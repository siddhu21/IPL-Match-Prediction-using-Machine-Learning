FROM python:3.8
ADD . /app
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY IPL_Data_with_First_Inns_Score.xlsx .
COPY model_22_03_ppw.pkl .
COPY static/images/ipl-2021-teams-ix7zwgff29ylomuf.jpg .

COPY . .

EXPOSE 5080
ENTRYPOINT [ "python" ]
CMD ["app.py"]

