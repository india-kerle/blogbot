from pydantic import BaseModel
from pydantic import Field

class BlogPost(BaseModel):
    """Structured format for blog posts."""
    text: str = Field(..., description="The content of the blog post.")