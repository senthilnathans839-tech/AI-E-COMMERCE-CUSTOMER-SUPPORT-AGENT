"""
tools.py
--------
Tool Calling layer for the AI E-Commerce Customer Support Agent.
Each function is exposed to the LLM as a callable "tool" using LangChain's
@tool decorator. In production, replace the JSON file reads with real
calls to your Order Management System / Product Catalog API / Returns API.
"""

import json
import os
from datetime import datetime, date
from langchain_core.tools import tool

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_json(filename):
    with open(os.path.join(DATA_DIR, filename), "r") as f:
        return json.load(f)


def _save_json(filename, data):
    with open(os.path.join(DATA_DIR, filename), "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------
# TOOL 1: Product search / catalog Q&A
# ---------------------------------------------------------------------
@tool
def search_product(query: str) -> str:
    """Search the product catalog by product name or category keyword.
    Returns matching products with price, stock availability and description.
    Use this when the customer asks about a product, its price, or if it's in stock.
    """
    products = _load_json("products.json")
    query_lower = query.lower()
    matches = [
        p for p in products
        if query_lower in p["name"].lower() or query_lower in p["category"].lower()
    ]
    if not matches:
        return f"No products found matching '{query}'."

    results = []
    for p in matches:
        stock_msg = f"In stock ({p['stock']} units)" if p["stock"] > 0 else "Out of stock"
        results.append(
            f"{p['name']} (ID: {p['product_id']}) - ₹{p['price']} - {stock_msg}\n"
            f"  {p['description']}\n"
            f"  Return window: {p['return_window_days']} days"
        )
    return "\n".join(results)


# ---------------------------------------------------------------------
# TOOL 2: Order status / tracking
# ---------------------------------------------------------------------
@tool
def check_order_status(order_id: str) -> str:
    """Look up the current status, tracking ID and expected delivery date
    for a given order ID (e.g. 'ORD5001'). Use this when a customer asks
    'where is my order' or 'when will it arrive'.
    """
    orders = _load_json("orders.json")
    order = next((o for o in orders if o["order_id"].upper() == order_id.upper()), None)
    if not order:
        return f"I couldn't find any order with ID '{order_id}'. Please double-check the order number."

    tracking = order["tracking_id"] or "Not yet assigned"
    return (
        f"Order {order['order_id']}: {order['product_name']} (x{order['quantity']})\n"
        f"Status: {order['status']}\n"
        f"Ordered on: {order['order_date']}\n"
        f"Expected delivery: {order['expected_delivery']}\n"
        f"Tracking ID: {tracking}"
    )


# ---------------------------------------------------------------------
# TOOL 3: List orders for a customer (used for account-level questions)
# ---------------------------------------------------------------------
@tool
def list_customer_orders(customer_email: str) -> str:
    """List all orders placed by a customer, given their email address.
    Use this when the customer doesn't remember their order ID.
    """
    orders = _load_json("orders.json")
    customer_orders = [o for o in orders if o["customer_email"].lower() == customer_email.lower()]
    if not customer_orders:
        return f"No orders found for {customer_email}."

    return "\n".join(
        f"{o['order_id']} - {o['product_name']} - {o['status']}" for o in customer_orders
    )


# ---------------------------------------------------------------------
# TOOL 4: Return eligibility check
# ---------------------------------------------------------------------
@tool
def check_return_eligibility(order_id: str) -> str:
    """Check whether an order is still eligible for return, based on the
    product's return window and delivery/order date. Use this before
    processing a return request.
    """
    orders = _load_json("orders.json")
    products = _load_json("products.json")

    order = next((o for o in orders if o["order_id"].upper() == order_id.upper()), None)
    if not order:
        return f"Order '{order_id}' not found."

    product = next((p for p in products if p["product_id"] == order["product_id"]), None)
    if not product:
        return "Product details unavailable for this order."

    if order["status"] != "Delivered":
        return f"Order {order_id} is currently '{order['status']}' — it must be delivered before a return can be initiated."

    order_date = datetime.strptime(order["order_date"], "%Y-%m-%d").date()
    days_elapsed = (date.today() - order_date).days
    window = product["return_window_days"]

    if days_elapsed <= window:
        return (
            f"Order {order_id} IS eligible for return. "
            f"{window - days_elapsed} day(s) remaining in the {window}-day return window."
        )
    else:
        return (
            f"Order {order_id} is NOT eligible for return. "
            f"The {window}-day return window closed {days_elapsed - window} day(s) ago."
        )


# ---------------------------------------------------------------------
# TOOL 5: Initiate a return (state-changing action)
# ---------------------------------------------------------------------
@tool
def initiate_return(order_id: str, reason: str) -> str:
    """Initiate a product return for an eligible order. Requires the order ID
    and a brief reason for the return. Always call check_return_eligibility
    first before using this tool.
    """
    orders = _load_json("orders.json")
    order = next((o for o in orders if o["order_id"].upper() == order_id.upper()), None)
    if not order:
        return f"Order '{order_id}' not found."

    if order["status"] == "Return Initiated":
        return f"A return for order {order_id} has already been initiated."

    order["status"] = "Return Initiated"
    order["return_reason"] = reason
    order["return_requested_on"] = date.today().isoformat()
    _save_json("orders.json", orders)

    return (
        f"Return initiated for order {order_id}. Reason: '{reason}'. "
        f"A pickup will be scheduled within 2-3 business days and refund will be "
        f"processed after quality check."
    )


# ---------------------------------------------------------------------
# TOOL 6: Escalate to a human agent
# ---------------------------------------------------------------------
@tool
def escalate_to_human(issue_summary: str, customer_email: str) -> str:
    """Escalate the conversation to a human support agent when the issue is
    complex, sensitive (e.g. fraud, damaged goods dispute), or the customer
    explicitly asks for a human. Provide a concise summary of the issue.
    """
    ticket_id = f"TCK{abs(hash(issue_summary + customer_email)) % 100000}"
    return (
        f"I've escalated this to our support team. Ticket ID: {ticket_id}. "
        f"A human agent will contact {customer_email} within 24 hours regarding: '{issue_summary}'."
    )


ALL_TOOLS = [
    search_product,
    check_order_status,
    list_customer_orders,
    check_return_eligibility,
    initiate_return,
    escalate_to_human,
]
