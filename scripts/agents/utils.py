def initialize_telemetry():
    """
    Initializes OpenTelemetry and Azure Monitor for telemetry.
    """
    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
    from azure.monitor.opentelemetry import configure_azure_monitor
    import logging
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)

    OpenAIInstrumentor().instrument()

    application_insights_connection_string = os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")
    if application_insights_connection_string:
        configure_azure_monitor(connection_string=application_insights_connection_string)
    else:
        logging.warning("APPLICATION_INSIGHTS_CONNECTION_STRING not set. Azure Monitor will not be configured.")