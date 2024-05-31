import json
import logging
import os
import traceback
from datetime import datetime
from decimal import Decimal as decimal

import boto3
from anthropic.types import ContentBlockDeltaEvent
from anthropic.types import Message as AnthropicMessage
from anthropic.types import MessageDeltaEvent, MessageStopEvent
from app.auth import verify_token
from app.bedrock import calculate_price, compose_args
from app.repositories.conversation import RecordNotFoundError, store_conversation
from app.repositories.models.conversation import ChunkModel, ContentModel, MessageModel
from app.routes.schemas.conversation import ChatInputWithToken
from app.usecases.bot import modify_bot_last_used_time
from app.usecases.chat import (
    get_bedrock_response,
    insert_knowledge,
    prepare_conversation,
    trace_to_root,
)
from app.utils import get_anthropic_client, get_current_time, is_anthropic_model
from app.vector_search import filter_used_results, search_related_docs
from boto3.dynamodb.conditions import Key
from ulid import ULID
from fastapi import WebSocket

WEBSOCKET_SESSION_TABLE_NAME = os.environ["WEBSOCKET_SESSION_TABLE_NAME"]

client = get_anthropic_client()
dynamodb_client = boto3.resource("dynamodb")
table = dynamodb_client.Table(WEBSOCKET_SESSION_TABLE_NAME)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

generating = {}

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection_id = websocket.headers.get("sec-websocket-key")

    generating[connection_id] = True
    try:
        while generating[connection_id]:
            data = await websocket.receive_text()
            response = await process_message(data)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        logger.info(f"User {connection_id} disconnected.")
    finally:
        generating.pop(connection_id, None)

async def process_message(message: str) -> str:
    await asyncio.sleep(1)  # Simulate an AI processing the message
    return f"Processed: {message}"

@socketio.on('stop_generation')
def handle_stop_generation(data):
    user_id = data['user_id']
    generating[user_id] = False
    emit('generation_stopped', {'user_id': user_id})

def handler(event, context):
    logger.info(f"Received event: {event}")
    route_key = event["requestContext"]["routeKey"]

    if route_key == "$connect":
        return {"statusCode": 200, "body": "Connected."}
    elif route_key == "$disconnect":
        return {"statusCode": 200, "body": "Disconnected."}

    connection_id = event["requestContext"]["connectionId"]
    domain_name = event["requestContext"]["domainName"]
    stage = event["requestContext"]["stage"]
    endpoint_url = f"https://{domain_name}/{stage}"
    gatewayapi = boto3.client("apigatewaymanagementapi", endpoint_url=endpoint_url)

    now = datetime.now()
    expire = int(now.timestamp()) + 60 * 2  # 2 minute from now
    body = event["body"]

    try:
        if body == "START":
            return {"statusCode": 200, "body": "Session started."}
        elif body == "END":
            # Concatenate the message parts and process the full message
            message_parts = []
            last_evaluated_key = None

            while True:
                if last_evaluated_key:
                    response = table.query(
                        KeyConditionExpression=Key("ConnectionId").eq(connection_id),
                        ExclusiveStartKey=last_evaluated_key,
                    )
                else:
                    response = table.query(
                        KeyConditionExpression=Key("ConnectionId").eq(connection_id)
                    )

                message_parts.extend(response["Items"])

                if "LastEvaluatedKey" in response:
                    last_evaluated_key = response["LastEvaluatedKey"]
                else:
                    break

            logger.info(f"Number of message chunks: {len(message_parts)}")
            full_message = "".join(item["MessagePart"] for item in message_parts)

            chat_input = ChatInputWithToken(**json.loads(full_message))
            return process_chat_input(chat_input, gatewayapi, connection_id)
        else:
            message_json = json.loads(body)
            part_index = message_json["index"]
            message_part = message_json["part"]

            table.put_item(
                Item={
                    "ConnectionId": connection_id,
                    "MessagePartId": decimal(part_index),
                    "MessagePart": message_part,
                    "expire": expire,
                }
            )
            return {"statusCode": 200, "body": "Message part received."}

    except Exception as e:
        logger.error(f"Operation failed: {e}")
        logger.error("".join(traceback.format_tb(e.__traceback__)))
        gatewayapi.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({"status": "ERROR", "reason": str(e)}).encode("utf-8"),
        )
        return {"statusCode": 500, "body": str(e)}
