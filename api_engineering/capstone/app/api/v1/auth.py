"""Auth routes: register, login (OAuth2 password flow), me."""
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm

from app import schemas
from app.api.deps import get_auth_service, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=schemas.UserRead, status_code=201)
async def register(data: schemas.RegisterRequest, service=Depends(get_auth_service)):
    # TODO: user = await service.register(data); return user
    user = await service.register(data)
    return user


@router.post("/login", response_model=schemas.Token)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    service=Depends(get_auth_service),
):
    """OAuth2 password flow: form.username carries the email, form.password the password.

    Stretch: apply the stricter login rate limit here (deps.limiter) to blunt
    credential stuffing.
    """
    # TODO: token = await service.authenticate(form.username, form.password)
    #       return schemas.Token(access_token=token, token_type="bearer")
    token = await service.authenticate(form.username, form.password)
    return schemas.Token(access_token=token, token_type="bearer")


@router.get("/me", response_model=schemas.UserRead)
async def me(user=Depends(get_current_user)):
    # TODO: return user
    return user