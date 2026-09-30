from agent_sdk import BaseAgent, calculate_metrics

def main():
    agent = BaseAgent(
        model_name="gpt-4o",
        tools=[calculate_metrics],
        system_prompt="You are an enterprise data assistant."
    )

    print("Agent initialized successfully!")
    # Example execution (requires active OPENAI_API_KEY)
    # response = agent.execute("Compute metrics for these numbers: [12.5, 45.0, 33.2]")
    # print("Response:", response)

if __name__ == "__main__":
    main()
