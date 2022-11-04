FROM python:3.7

ENV TZ=Asia/Shanghai

WORKDIR /root/mgs-core
COPY requirements.txt requirements.txt
RUN pip3 install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . .

CMD ["python3", "server/Server.py"]
