FROM python:3.6
COPY . /
WORKDIR /
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["app.py"]