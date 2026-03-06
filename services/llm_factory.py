import os

def get_llm(temperature=0):
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "openai":
        from langchain.chat_models import ChatOpenAI
        model_name = os.getenv("LLM_MODEL", "gpt-3.5-turbo-1106")
        return ChatOpenAI(temperature=temperature, model=model_name)
        
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            try:
                from langchain.chat_models import ChatAnthropic
            except ImportError:
                raise ImportError("Please install langchain-anthropic to use Anthropic models: pip install langchain-anthropic")
        model_name = os.getenv("LLM_MODEL", "claude-3-sonnet-20240229")
        return ChatAnthropic(temperature=temperature, model=model_name)
        
    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Please install langchain-google-genai to use Google models: pip install langchain-google-genai")
        
        # Default to 2.5 Flash as requested, fallback to 1.5 if environment varies
        model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(temperature=temperature, model=model_name)
        
    elif provider == "mistral":
        try:
            from langchain_mistralai.chat_models import ChatMistralAI
        except ImportError:
            raise ImportError("Please install langchain-mistralai to use Mistral models: pip install langchain-mistralai")
        model_name = os.getenv("LLM_MODEL", "mistral-large-latest")
        return ChatMistralAI(temperature=temperature, model=model_name)
        
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

def get_llm_with_json(llm):
    """
    Attempts to bind JSON response format if supported by the provider.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "openai":
        return llm.bind(response_format={"type": "json_object"})
    elif provider == "mistral":
        return llm.bind(response_format={"type": "json_object"})
    elif provider == "google":
        # Gemini supports native JSON schema/mime-type in modern versions
        return llm.bind(response_mime_type="application/json")
    
    return llm
