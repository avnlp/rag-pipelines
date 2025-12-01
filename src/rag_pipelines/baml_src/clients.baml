// Retry Policy for the clients
retry_policy Exponential {
  max_retries 2
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}

client<llm> GroqClient {
  provider openai-generic
  retry_policy Exponential
  options {
    base_url "https://api.groq.com/openai/v1"
    api_key env.GROQ_API_KEY
    model "qwen/qwen3-32b"
    temperature 0.6
    max_tokens 4096
    top_p 0.95
  }
}

client<llm> CerebrasClient {
  provider "openai-generic"
  retry_policy Exponential
  options {
    base_url "https://api.cerebras.ai/v1"
    api_key env.CEREBRAS_API_KEY
    model "qwen-3-32b"
    temperature 0.6
    max_tokens 4096
    top_p 0.95
  }
}

client<llm> SambaNovaClient {
  provider "openai-generic"
  retry_policy Exponential
  options {
    base_url "https://api.sambanova.ai/v1"
    api_key env.SAMBANOVA_API_KEY
    model "qwen-3-32b"
    temperature 0.6
    max_tokens 4096
    top_p 0.95
  }
}


// Fallback Client - Tries Groq, then Cerebras, then SambaNova
client<llm> GroqFallbackClient {
  provider fallback
  options {
    // This will try the clients in order until one succeeds
    // If Groq fails, try Cerebras, then SambaNova
    strategy [GroqClient, CerebrasClient, SambaNovaClient]
  }
}

