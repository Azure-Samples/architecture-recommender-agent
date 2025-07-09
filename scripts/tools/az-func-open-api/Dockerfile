# To enable ssh & remote debugging on app service change the base image to the one below
# FROM mcr.microsoft.com/azure-functions/python:4-python3.12-appservice
FROM mcr.microsoft.com/azure-functions/python:4-python3.12

# To test in local docker we can pass in the environment variables
ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true \
    AzureWebJobsStorage="UseDevelopmentStorage=true" \
    FUNCTIONS_WORKER_RUNTIME=python \
    AZURE_OPENAI_ENDPOINT= \
    AZURE_OPENAI_KEY= \
    AZURE_OPENAI_EMBEDDINGS_MODEL_NAME=text-embedding-3-large \
    AZURE_AI_SEARCH_ENDPOINT= \
    AZURE_AI_SEARCH_KEY= \
    AZURE_AI_SEARCH_INDEX_NAME=cw-architectures-index \
    STORAGE_CONNECTION= \
    WEBSITE_HOSTNAME=localhost:8080

# Expose the default Azure Functions port
EXPOSE 80

COPY requirements.txt /
RUN pip install -r /requirements.txt

COPY . /home/site/wwwroot