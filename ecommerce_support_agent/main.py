"""
main.py
-------
Simple command-line chat loop to test the AI E-Commerce Customer Support Agent.

Usage:
    python main.py
"""

from agent import EcommerceSupportAgent


def main():
    print("=" * 60)
    print(" ShopAssist — AI E-Commerce Customer Support Agent")
    print(" (Tool Calling + Memory, powered by an open-source LLM)")
    print("=" * 60)

    email = input("Enter your registered email to start: ").strip() or "arun@example.com"
    agent = EcommerceSupportAgent(customer_email=email)

    print(f"\nHi! I'm ShopAssist. How can I help you today, {email}?")
    print("(type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("ShopAssist: Thanks for reaching out! Have a great day. 👋")
            break
        if not user_input:
            continue

        reply = agent.chat(user_input)
        print(f"ShopAssist: {reply}\n")


if __name__ == "__main__":
    main()
