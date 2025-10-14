from pydantic import BaseModel, EmailStr, Field
from typing import List


class CreateUser(BaseModel):
    user_name: str = Field(..., max_length=255, example="JohnDoe")
    email: EmailStr = Field(..., example="johndoe@example.com")
    password: str = Field(..., min_length=6, example="securePassword123")


class GetUser(BaseModel):
    user_name: str
    email: EmailStr


class GetUsers(BaseModel):
    users: List[GetUser]
