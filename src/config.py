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
    FOUNDRY_AGENT_ID = os.environ.get("FOUNDRY_AGENT_ID", "")
    FOUNDRY_PROJECT_ENDPOINT = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "")
    FOUNDRY_PROJECT_KEY = os.environ.get("FOUNDRY_PROJECT_KEY", "")
    AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
    AZURE_RESOURCE_GROUP_NAME = os.environ.get("AZURE_RESOURCE_GROUP_NAME", "")
    DUMMY_USER_ENTRA_ID = os.environ.get("DUMMY_USER_ID", "")