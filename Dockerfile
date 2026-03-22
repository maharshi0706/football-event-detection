FROM python:3.13.12

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.6.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu    

COPY requirements_no_torch.txt .
RUN pip install --no-cache-dir -r requirements_no_torch.txt

COPY Api/ ./Api/
COPY Inference/ ./Inference/

EXPOSE 7860

CMD [ "uvicorn", "Api:main:app", "--host", "0.0.0.0", "--port", "7860" ]