import os
from pathlib import Path
from dotenv import load_dotenv
from pdf2image import convert_from_path
import fitz
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import AzureOpenAI
import json
import base64
import fitz  # PyMuPDF
from pathlib import Path
from typing import Tuple
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex
)
from azure.core.credentials import AzureKeyCredential, AccessToken
import uuid
from azure.identity import ClientSecretCredential
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from msal import ConfidentialClientApplication
#import logging
#logging.basicConfig(level=logging.DEBUG)

# Set up paths
script_dir = Path.cwd()
data_dir = script_dir / "data"          # …/data
out_page_dir  = data_dir / "split_pages"      # …/data/split_pages
out_page_dir.mkdir(parents=True, exist_ok=True)
out_fig_dir = data_dir / "figures"
out_fig_dir.mkdir(parents=True, exist_ok=True)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

#configure service connections
endpoint = os.environ["Azure_Document_Intelligence_Endpoint"]#
key       = os.environ["Azure_Document_Intelligence_Key"]#
adi_client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

aoai_endpoint   = os.environ["AZURE_OPENAI_ENDPOINT"]#
aoai_key        = os.environ["AZURE_OPENAI_KEY"]#
aoai_deployment = "gpt-4o-2"                      
api_version     = "2024-12-01-preview"  

aoai_client = AzureOpenAI(
    api_key=aoai_key,
    azure_endpoint=aoai_endpoint,
    api_version=api_version,
)

search_endpoint = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
search_admin_key = os.environ["AZURE_AI_SEARCH_KEY"]
index_name = os.environ["AZURE_AI_SEARCH_INDEX_NAME"]
embedding_deployment = os.environ["Azure_OpenAI_Embedding_Deployment_Name"]
vector_config = "arch-hnsw"
azure_openai_embedding_dimensions = 3072

cred          = AzureKeyCredential(search_admin_key)
index_client  = SearchIndexClient(search_endpoint, cred)

# Blob Service Principal credentials
#tenant_id = os.environ["Azure_Blob_SP_Tenant_Id"] 
#client_id = os.environ["Azure_Blob_SP_Client_Id"]
#client_secret = os.environ["Azure_Blob_SP_Client_Secret"]
#authority = f"https://login.microsoftonline.com/{tenant_id}"
#scope = ["https://storage.azure.com/.default"]

# Create a Confidential Client Application
#app = ConfidentialClientApplication(
#    client_id=client_id,
#    client_credential=client_secret,
#    authority=authority
#)
# Acquire a Token
#token_response = app.acquire_token_for_client(scopes=scope)
#if "access_token" in token_response:
#    access_token = token_response["access_token"]
#else:
#    raise Exception("Failed to acquire token")

# Wrap the access token
class BearerTokenCredential:
    def __init__(self, token):
        self.token = token

    def get_token(self, *scopes):
        return AccessToken(self.token, float("inf"))

#credential = BearerTokenCredential(access_token)

# Azure Storage account details
#storage_account_name = os.environ["Azure_Blob_Storage_Account_Name"]
#container_name = os.environ["Azure_Blob_Container_Name"]
#blob_service_client = BlobServiceClient(
#    account_url=f"https://{storage_account_name}.blob.core.windows.net",
#    credential=credential
#)

architecture_extraction_system_prompt = """
You are provided with the OCR content and Section Headings of a PDF containing software architecture diagrams. Your job is to use the Sections Headings of the PDF to identify 
all the different services being used in the architecture diagrams. In the full page ocr content, the section heading for an architecture diagram will always
be followed by the architecture diagram which will contain the different services being used in a particular workflow. 
Use the section headings to identify the beginning of each diagram in the full ocr content. Not all section headings will be for architecture diagrams and some will be for  supplemental content. 
Only extract the service names from the contents of each architecture diagram and ignore any content used to describe the workflow.
Split the service names between Azure and non-Azure services. 
Note: Azure Cloud is not a service, it is a cloud platform. Do not include it in the list of services.

Output Format: Must strictly adhere to the json schema. 
"""

system_prompt_arch_summary = """
You will receive an image that contains one or more software‑architecture diagrams.
Your tasks:
1. Detect every architecture diagram in the image (usually separated by a visible title).
2. For each diagram, produce an AI summary that includes:
   • Architecture Name (exact title text you see).
   • Detailed explaination of the workflow of the architecture diagram. 

Format:
Architecture Name: <title>
Summary: <detailed explaination>

Return the summaries in the order the diagrams appear (top‑to‑bottom, left‑to‑right).
If no diagram or services are detected, reply “No architecture diagram recognized”.
"""

class ArchitecutreSchema(BaseModel):
    name: str
    azure_services: str
    non_azure_services: str

class ArchitectureExtraction(BaseModel):
    extracted_architectures: List[ArchitecutreSchema]


class ArchitecutreImagesSchema(BaseModel):
    name: str
    summary: str

class ArchitectureAISummaries(BaseModel):
    extracted_architecture_summaries: List[ArchitecutreImagesSchema]


def create_or_update_search_index() -> None:

    try: 
        index_client.get_index(index_name)
        print(f"Index {index_name} already exists")
    except:
        print(f"Creating index {index_name}")

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="name", type=SearchFieldDataType.String, searchable=True, filterable=True),
            SimpleField(name="architecture_url", type=SearchFieldDataType.String, searchable=True, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="myHnswProfile"
        )]

        vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            )
        ])
        
        semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="name"),
            content_fields=[SemanticField(field_name="content")]
        ))

        semantic_search = SemanticSearch(configurations=[semantic_config])
        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
        index_result = index_client.create_or_update_index(index)
        print(f'{index_result.name} created')


def get_ocr_from_adi(file_path: str):
    
    with open(file_path, "rb") as f:
        poller = adi_client.begin_analyze_document(
       "prebuilt-layout",
        body=f)

    result = poller.result()

    section_headings = []
    for paragraph in result.paragraphs:
        if getattr(paragraph, "role", None) == "sectionHeading":
            section_headings.append(paragraph.content)

    section_heading_text = "\n".join(section_headings) 

    fig_bounding_boxes = []
    for fig in (getattr(result, "figures", None) or []):
        caption = getattr(fig, "caption", None)
        bounding_box = fig["boundingRegions"][0]
        fig_bounding_boxes.append(bounding_box)
        if caption and getattr(caption, "content", None):
            section_headings.append(caption.content)

    return section_headings, fig_bounding_boxes, result


def _to_points(poly) -> list[fitz.Point]:
    """
    Accepts either a flat list [x1,y1,x2,y2,x3,y3,x4,y4] or
    a list of 4 (x,y) tuples and returns a list[fitz.Point].
    """
    if isinstance(poly, (list, tuple)) and len(poly) == 8 and all(isinstance(v, (int, float)) for v in poly):
        # flatten → pairs
        poly = list(zip(poly[0::2], poly[1::2]))
    if len(poly) != 4:
        raise ValueError("polygon must contain four points")
    return [fitz.Point(x * 72, y * 72) for x, y in poly]

def pdf_to_figures(pdf_path: Path, out_fig_dir: Path, dpi: int = 300) -> None:
    """
    Extract figures from each page of the pdf and save them in the specified directory.
    """
    doc  = fitz.open(pdf_path)
    zoom = dpi / 72.0                       
    mat  = fitz.Matrix(zoom, zoom)

    for fig_idx,fig_bounding_box in enumerate(fig_bounding_boxes):
        quad = _to_points(fig_bounding_box["polygon"])
        page = doc[fig_bounding_box["pageNumber"]-1]
        xs, ys = zip(*quad)
        bbox = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
        zoom = dpi / 72.0
        mat  = fitz.Matrix(zoom, zoom)
        pix  = page.get_pixmap(matrix=mat, clip=bbox, alpha=False)
        out_fig_path = out_fig_dir / f"{pdf_path.stem}_{fig_idx:03}.png"
        pix.save(out_fig_path)


def pdf_to_pngs(pdf_path: Path, out_fig_dir: Path, out_page_dir: Path, dpi: int = 300) -> None:

    doc  = fitz.open(pdf_path)
    zoom = dpi / 72.0                       
    mat  = fitz.Matrix(zoom, zoom)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        w, h = page.rect.width, page.rect.height
        pix = doc.load_page(page_idx).get_pixmap(matrix=mat, alpha=False)
        out_page_path = out_page_dir / f"{pdf_path.stem}_{page_idx:03}.png"
        pix.save(out_page_path)
        
    
    pdf_to_figures(pdf_path, out_fig_dir, dpi=dpi)


def architecture_extraction_with_ocr(ocr_content: str, section_headings: str, architecture_extraction_system_prompt: str):
    """
    Extract architecture information from the OCR content and section headings.
    """

    user_message = (
    "Section Headings of PDF:\n"
    f"{section_headings}\n\n"
    "Full OCR Content:\n"
    f"{ocr_content}")

    extracted_architectures = []

    response = aoai_client.beta.chat.completions.parse(
    model=aoai_deployment,          # deployment‑level model name
    messages=[
        {"role": "system", "content": architecture_extraction_system_prompt},
        {"role": "user",   "content": user_message},
    ],
    temperature=0.2,
    max_tokens=2500,
    response_format=ArchitectureExtraction
)

    response = json.loads(response.choices[0].message.content)
    extracted_architectures.extend(response["extracted_architectures"])
    return extracted_architectures


def architecture_ai_summaries_with_images(pdf_path: Path, file_name: str, system_prompt_arch_summary: str):

    doc  = fitz.open(pdf_path)

    architecture_ai_summaries = []                                              

    for page_idx in range(len(doc)):                        
        image_path = out_page_dir / f"{Path(file_name).stem}_{page_idx:03}.png"
        print("Image Path:", image_path)
        if not image_path.exists():
            print(f"File missing: {image_path}")
            continue
        
        with open(image_path, "rb") as img_f:
            #blob_client.upload_blob(img_f, overwrite=True)
            image_base64 = base64.b64encode(img_f.read()).decode("utf-8")

        messages = [
            {"role": "system", "content": system_prompt_arch_summary},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the architecture diagram image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            },
        ]

        response = aoai_client.beta.chat.completions.parse(
            model=aoai_deployment,
            messages=messages,
            temperature=0.2,
            max_tokens=2500,
            response_format=ArchitectureAISummaries
        )

        reply = json.loads(response.choices[0].message.content)
        architecture_ai_summaries.extend(reply["extracted_architecture_summaries"])
    return architecture_ai_summaries


def build_and_push_docs(arch_items: List[dict], summaries: List[dict], file_name: str) -> None:
    summary_map = {s["name"]: s["summary"] for s in summaries}
    docs = []

    for index, arch in enumerate(arch_items):
        # Get a BlobClient
        blob_name = f"{Path(file_name).stem}_{index:03}.png"
        blob_path = str(out_fig_dir / blob_name)
        #blob_client = blob_service_client.get_blob_client(container=container_name, blob=f"{blob_path}_{index:03}.png")
        
        #container_client = blob_service_client.get_container_client(container=container_name)

        #with open(blob_path, "rb") as data:
        #    container_client.upload_blob(name=blob_name, data=data, overwrite=True)
            #blob_client.upload_blob(data, overwrite=True)

        #blob_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{blob_name}"
        blob_url = "https://example.blob.core.windows.net/container/"  # Replace with actual blob URL
        print(f"Blob uploaded successfully. URL: {blob_url}")

        text = (
            f"{arch['name']}. "
            f"Azure services: {arch['azure_services']}. "
            f"Non‑Azure services: {arch['non_azure_services']}. "
            f"AI Summary: {summary_map.get(arch['name'], '')}"
        )

        emb = aoai_client.embeddings.create(
            model=embedding_deployment,
            input=[text],
        ).data[0].embedding

        docs.append(
            {
                "id": str(uuid.uuid4()),
                "name": arch["name"],
                "content": text,
                "content_vector": emb,
                "architecture_url": blob_url,
            }
        )
    
    search_client = SearchClient(search_endpoint, index_name, cred)
    upload_result = search_client.upload_documents(docs)
    print("Upload succeeded:", all(r.succeeded for r in upload_result))

if __name__ == "__main__":
    print("Creating or updating search index...")
    create_or_update_search_index()
    print("Beginning data pipeline...")
    print(f"Data directory: {data_dir}")
    print(f"Script Directory: {script_dir}")
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            print(f"Processing file: {file_name}")
            file_path = data_dir / file_name
            section_headings, fig_bounding_boxes, result = get_ocr_from_adi(str(file_path))
            pdf_to_pngs(file_path, out_fig_dir, out_page_dir, dpi=300)
            extracted_architectures = architecture_extraction_with_ocr(
            ocr_content=result.content,
            section_headings=section_headings,
            architecture_extraction_system_prompt=architecture_extraction_system_prompt
            )
            architecture_ai_summaries = architecture_ai_summaries_with_images(
            pdf_path=file_path,
            file_name=file_name,
            system_prompt_arch_summary=system_prompt_arch_summary
            )
            build_and_push_docs(
            arch_items=extracted_architectures,
            summaries=architecture_ai_summaries, 
            file_name=file_name
            )
