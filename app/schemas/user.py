from pydantic import BaseModel, EmailStr, Field
from typing import List
from uuid import UUID
from datetime import datetime


class CreateUser(BaseModel):
    user_name: str = Field(..., max_length=255, example="JohnDoe")
    email: EmailStr = Field(..., example="johndoe@example.com")
    password: str = Field(..., min_length=6, example="securePassword123")


class GetUser(BaseModel):
    user_name: str
    email: EmailStr


class GetUsers(BaseModel):
    users: List[GetUser]


class LoginUser(BaseModel):
    email: EmailStr = Field(..., example="johndoe@example.com")
    password: str = Field(..., min_length=6, example="securePassword123")


class UploadedFileSchema(BaseModel):
    id: UUID
    original_filename: str
    unique_filename: str
    created_at: datetime


class UserUploadsResponse(BaseModel):
    status_code: int
    uploads: List[UploadedFileSchema]
