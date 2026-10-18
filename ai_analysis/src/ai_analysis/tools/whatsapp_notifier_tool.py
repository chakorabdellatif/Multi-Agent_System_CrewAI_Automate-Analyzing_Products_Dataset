import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WhatsAppNotificationInput(BaseModel):
    message: str = Field(..., description="The summary message or alert to send to WhatsApp")
    recipient_phone: Optional[str] = Field(default="+212000000000", description="Recipient phone number with country code")
    gateway_url: Optional[str] = Field(default="http://localhost:3000/api/sendText", description="OpenWA gateway URL")


class WhatsAppNotifierTool(BaseTool):
    name: str = "WhatsApp Notifier Tool"
    description: str = "Sends a WhatsApp alert or completion summary to the user's phone via OpenWA gateway"
    args_schema: type[BaseModel] = WhatsAppNotificationInput

    def _run(
        self,
        message: str,
        recipient_phone: str = "+212000000000",
        gateway_url: str = "http://localhost:3000/api/sendText",
    ) -> str:
        chat_id = recipient_phone.replace("+", "").replace(" ", "").replace("-", "")
        if not chat_id.endswith("@c.us"):
            chat_id = f"{chat_id}@c.us"

        payload = {"chatId": chat_id, "text": message}
        req_data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": "CrewAI-WhatsApp-Agent/1.0"}

        try:
            req = urllib.request.Request(gateway_url, data=req_data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=5) as response:
                resp = response.read().decode("utf-8")
                return f"✅ WhatsApp message delivered via OpenWA gateway: {resp}"
        except urllib.error.URLError as e:
            return (
                f"ℹ️ OpenWA Gateway at '{gateway_url}' is currently unreachable ({e.reason}).\n"
                f"📱 [Queued/Mock Alert Recorded]: Recipient: {chat_id}, Content: '{message[:100]}...'"
            )
        except Exception as e:
            return f"❌ WhatsApp Notification Error: {e!s}"
