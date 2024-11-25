FROM ubuntu

LABEL version="0.2.1"

RUN apt update
RUN apt upgrade -y
RUN apt install -y python3 python3-pip libpq-dev

EXPOSE 8080

RUN useradd --system --shell /bin/bash --uid 1001 -m --no-log-init ics

RUN mkdir -p /ics/db
RUN mkdir -p /ics/data
RUN mkdir -p /ics/log
RUN chown -R ics /ics

USER ics
WORKDIR /ics

ADD ics ics
ADD requirements.txt .

RUN pip3 install -r requirements.txt
RUN pip3 install psycopg2

VOLUME /ics/db
VOLUME /ics/data
VOLUME /ics/log

ENV DB_CONNECTION_STRING=sqlite:////ics/db/ics.sqlite

CMD PYTHONPATH=. python3 ics/scripts/webapp.py --host 0.0.0.0 --data_dir /ics/data --log_dir /ics/log --db_connection_string $DB_CONNECTION_STRING
