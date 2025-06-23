"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import os

from dotenv import load_dotenv

load_dotenv()

class Config:
    """Bot Configuration"""

    PORT = 3978
    APP_ID  = os.environ.get("APP_ID", "")
    APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
    APP_TYPE = os.environ.get("APP_TYPE", "")
    APP_TENANTID = os.environ.get("APP_TENANTID", "")
    FOUNDRY_AGENT_ID = os.environ.get("FOUNDRY_AGENT_ID", "asst_WvDPp6FuyrwLO3AWisHp1Mdp")
    FOUNDRY_PROJECT_ENDPOINT = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "https://mumann-05292025-resource.services.ai.azure.com/api/projects/mumann-05292025")
    FOUNDRY_PROJECT_KEY = os.environ.get("FOUNDRY_PROJECT_KEY", "2AEkcZAYry8koC50DkKMGk6rWj10fim5Ab4Q7TEq9nxlDEd22Xz8JQQJ99BEACHYHv6XJ3w3AAAAACOGItgH")
