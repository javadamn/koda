# llm_setup_eval.py
import os
import config
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from typing import Optional # Make sure Optional is imported

# Placeholder for a LangChain DeepSeek wrapper if available and preferred
# from langchain_community.chat_models import ChatDeepseek

logger = config.get_logger(__name__)

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
# OPENAI_API_KEY is often picked up automatically by ChatOpenAI if set in env
# GOOGLE_APPLICATION_CREDENTIALS for Vertex AI (usually set for gcloud)

LLM_PROVIDER_PREFIXES = {
    "gpt": "openai",
    "claude": "anthropic",
    "gemini": "google_vertexai",
    "deepseek": "deepseek"  # This is a key for logic
}

def get_llm_provider(model_name: str) -> str:
    model_name_lower = model_name.lower()
    for prefix, provider in LLM_PROVIDER_PREFIXES.items():
        if model_name_lower.startswith(prefix):
            return provider
    logger.warning(f"Could not determine provider for model: {model_name}. Defaulting to 'openai'.")
    return "openai"

# Ensure this signature is exactly as follows:
def get_llm(model_name: str, temperature: Optional[float] = None): # Accepts parameters
    """
    Gets a LangChain LLM instance by model name, handling different providers.
    """
    provider = get_llm_provider(model_name)
    
    final_temperature = temperature if temperature is not None \
        else getattr(config, 'LLM_TEMPERATURE', 0.7) # Get from config or default to 0.7

    logger.info(f"Initializing LLM: Model='{model_name}', Provider='{provider}', Temp='{final_temperature}'")

    if provider == "openai":
        return ChatOpenAI(
            model_name=model_name,
            temperature=final_temperature,
            # openai_api_key=os.environ.get("OPENAI_API_KEY") # Usually auto-detected
        )
    elif provider == "anthropic":
        api_key = os.environ.get(ANTHROPIC_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"{ANTHROPIC_API_KEY_ENV} not found for Anthropic model {model_name}.")
        return ChatAnthropic(
            model=model_name, # For ChatAnthropic, the parameter is 'model'
            temperature=final_temperature,
            anthropic_api_key=api_key
        )
    elif provider == "google_vertexai":
        # Ensure GOOGLE_APPLICATION_CREDENTIALS env var is set for service account auth,
        # or `gcloud auth application-default login` has been run.
        return ChatVertexAI(
            model_name=model_name,
            temperature=final_temperature,
            # project="your-gcp-project-id", # Add if needed, or get from config
            # location="your-gcp-region",   # Add if needed, or get from config
        )
    elif provider == "deepseek":
        # ** YOU MUST IMPLEMENT THIS SECTION BASED ON DEEPSEEK'S LANGCHAIN INTEGRATION **
        # Option A: If DeepSeek has an OpenAI-compatible endpoint
        # deepseek_api_key = os.environ.get(DEEPSEEK_API_KEY_ENV)
        # deepseek_base_url = os.environ.get("DEEPSEEK_API_BASE_URL") # Ensure this ENV VAR is set
        # if not deepseek_api_key:
        #     raise ValueError(f"{DEEPSEEK_API_KEY_ENV} not found for DeepSeek model {model_name}.")
        # if not deepseek_base_url:
        #     raise ValueError(f"DEEPSEEK_API_BASE_URL not found for DeepSeek model {model_name}.")
        # logger.info(f"Using OpenAI-compatible API for DeepSeek model: {model_name} at {deepseek_base_url}")
        # return ChatOpenAI(
        #     model_name=model_name, # The specific model identifier for DeepSeek at that endpoint
        #     temperature=final_temperature,
        #     api_key=deepseek_api_key, 
        #     base_url=deepseek_base_url 
        # )

        # Option B: If there's a dedicated LangChain DeepSeek class
        # from langchain_community.chat_models import ChatDeepseek # Replace with actual import
        # deepseek_api_key = os.environ.get(DEEPSEEK_API_KEY_ENV)
        # if not deepseek_api_key:
        #     raise ValueError(f"{DEEPSEEK_API_KEY_ENV} not found for DeepSeek model {model_name}.")
        # return ChatDeepseek(model=model_name, deepseek_api_key=deepseek_api_key, temperature=final_temperature)
        
        logger.error(f"DeepSeek provider for model '{model_name}' is not fully configured in get_llm. Please implement.")
        raise NotImplementedError(f"LLM provider 'deepseek' for model '{model_name}' not implemented yet. Update llm_setup_eval.py with its LangChain integration.")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider} for model {model_name}")