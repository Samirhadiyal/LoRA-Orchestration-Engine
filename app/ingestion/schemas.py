from pydantic import BaseModel


class ParsedPage(BaseModel):
    text: str
    page_number: int