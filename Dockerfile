FROM python:3.7

ENV TZ=Asia/Shanghai

WORKDIR /root/mgs-core
RUN python -m pip install --upgrade pip
RUN pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt -i
COPY . .

CMD ["python", "-m", "src.rpc.server"]
