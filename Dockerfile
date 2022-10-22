FROM python:3.7

ENV TZ=Asia/Shanghai

WORKDIR /root/mgs-core
COPY . .

CMD ["python3", "server/Server.py"]