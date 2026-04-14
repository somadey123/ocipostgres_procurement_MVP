from typing import Optional

from pydantic import BaseModel


class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None
