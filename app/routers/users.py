from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.user import CreateUser, GetUser, GetUsers, LoginUser
from app.dependencies import get_db, get_current_user
from app.models.users import Users
from sqlalchemy import select
from app.utils.password import hash_password, verify_password
from app.utils.tokens import create_access_token, create_refresh_token
from app.tasks.email import send_email_task

router = APIRouter(prefix="/user", tags=["User"])


@router.post("/signup")
async def signup(user: CreateUser, db: AsyncSession = Depends(get_db)):
    existance = await db.execute(select(Users).where(Users.email == user.email))
    exist = existance.scalar_one_or_none()
    if exist:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )
    password = hash_password(user.password)

    new_user = Users(username=user.user_name, email=user.email, password=password)
    db.add(new_user)
    await db.commit()

    subject = "Testing - Signup Successful"
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Welcome!</title>
    </head>
    <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
        <div style="max-width: 600px; margin: auto; background-color: white; padding: 30px; border-radius: 10px; text-align: center;">
            <h1 style="color: #4CAF50;">Welcome to the Team!</h1>
            <p>Hi there,</p>
            <p>We are thrilled to have you on board. Get ready for an exciting journey!</p>
            <p>Best regards,<br>Your Company Name</p>
        </div>
    </body>
    </html>
    """
    send_email_task.delay(user.email, subject, html_content)
    return {
        "status_code": status.HTTP_201_CREATED,
        "message": "User registered successfully",
    }


@router.post("/login")
async def login(data: LoginUser, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Users).where(Users.email == data.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(data.password, user.password):
        return {
            "status_code": status.HTTP_404_NOT_FOUND,
            "message": "Invalid email or password",
        }
    access_token = create_access_token({"sub": str(user.id), "email": user.email})
    refresh_token = create_refresh_token({"sub": str(user.id)})
    return {
        "status_code": status.HTTP_200_OK,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "message": "LoggedIn Successfully",
    }


@router.get("/getuser")
async def user_details(current_user: Users = Depends(get_current_user)):
    return {
        "status_code": status.HTTP_200_OK,
        "id": current_user.id,
        "user_name": current_user.username,
        "email": current_user.email,
    }
