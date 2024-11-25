# ICS - Interactive Classification System

The Interactive Classification System (ICS), is a web-based application that supports the activity
of manual text classification, i.e., labeling documents according to their content.

The system is designed to give total freedom of action to its users: they can at any time modify any classification
schema and any label assignment, possibly reusing any relevant information from previous activities.

The application uses machine learning to actively support its users with classification suggestions The machine learning
component of the system is an unobtrusive observer of the users' activities, never interrupting them, constantly
adapting and updating its models in response to their actions, and always available to perform automatic
classifications.

* [Publication](#publication)
* [Installation](#installation)
* [Starting the main app](#startmain)
* [Login](#login)
* [Configuration](#configuration)
* [Additional apps](#apps)
* [Video tutorials](#videos)
* [License](#license)

## <a name="publication"></a> Publication

ICS is described in the paper:

[A. Esuli, "ICS: Total Freedom in Manual Text Classification Supported by Unobtrusive Machine Learning," in IEEE Access, vol. 10, pp. 64741-64760, 2022, doi: 10.1109/ACCESS.2022.3184009](https://doi.org/10.1109/ACCESS.2022.3184009)

## <a name="installation"></a> Installation

You can have a working installation of ICS in many ways:

- [Single file executable](#single-file-executable) (to start using ICS)
- [Docker](#docker) (for a single user)
- [Docker compose](#docker-compose) (for larger installation)
- [Pip install](#pip)
- [From source](#from-source)

### Single file executable

Executable files of ICS are downloadable from the [releases page](https://github.com/aesuli/ics/releases).
Once downloaded it can be run and have a working instance of ICS, provided a [database is configured](#db-configuration).

```shell
ics-webapp
```


The executable are from source using [pyinstaller](https://pyinstaller.org/):

```shell
pyinstaller -F ics\scripts\webapp.py --add-data="ics\apps\media;ics\apps\media" --collect-all sklearn --name ics-webapp
```

### Docker

A quick way have a running instance of ICS is to use [Docker](https://docker.com).

```shell
docker run -p 8080:8080 ghcr.io/aesuli/ics
```

This command pulls the ICS image from Docker hub and runs it, publishing the application on port 8080 of the host machine, accessible from any interface.
Once started ICS is accessible from the host machine using a browser at the address [http://127.0.0.1:8080](http://127.0.0.1:8080)

To have ICS accessible only from the local host machine add local ip address:

```shell
docker run -p 127.0.0.1:8080:8080 ghcr.io/aesuli/ics
```

__NOTE:__ by default the ICS image uses the SQLite database engine, which may result in reduced efficiency and functionalities.
A configuration using PostgreSQL is strongly recommended. It can be easily set up using docker compose.

#### Data persistence

ICS image use volumes to keep information persistent:
- ics-db stores the sqlite file, this is the only volume that should be saved to keep the state of the application.
- ics-data stores the files that are uploaded or downloaded from the system. It is defined for inspection in case of failures, it is not necessary to save it.
- ics-log stores the log files. It is defined for inspection in case of failures, it is not necessary to save it.

### Docker compose

An instance of ICS using PostgreSQL can be obtained downloading the [docker-compose.yml](https://github.com/aesuli/ics/raw/main/docker-compose.yml) file to a local directory and running

```shell
docker compose up
```
from that directory.

#### Host and port

The environment variables ``ICS_HOST`` and ``ICS_PORT`` define the interface and port on which ICS is accessible on the host machine.
Default is 127.0.0.1 and 8080.

#### Data persistence

The compose-based version of ICS use volumes to keep information persistent:
- db-data stores the PostgreSQL, this is the only volume that should be saved to keep the state of the application.
- ics-data stores the files that are uploaded or downloaded from the system. It is defined for inspection in case of failures, it is not necessary to save it.
- ics-log stores the log files. It is defined for inspection in case of failures, it is not necessary to save it.

A volume can be linked to a path on the host machine by defining an environment variable (or by editing the docker-compose.yml file):
- DB_DATA for the db-data volume (recommended)
- ICS_DATA for the ics-data volume (not necessary)
- ICS_LOG for the ics-log volume (not necessary)

For example, on Windows:
```shell
set DB_DATA=D:\ics_db_data
docker compose up
```

On Linux/Mac:
```shell
DB_DATA=/var/lib/ics/data docker compose up
```

### Pip

The suggested way to quickly set up the python environment is to use
the [Anaconda/Miniconda distribution](https://www.anaconda.com/products/distribution) and the `conda` package manager to
create the virtual enviroment.

```shell
conda create -n ics python
conda activate ics
````

ICS is published as a [`pip` package](https://pypi.org/project/ics-pkg).

```shell
pip install ics-pkg
```

The last required step is to [configure a database](#db-configuration).

### From source

Download source code from [GitHub repo](https://github.com/aesuli/ics).
Create a virtual environment and install the required packages.

```shell
cd [directory with ICS code]
conda create -n ics python
conda activate ics
pip install -r requirements.txt
```

The last required step is to [configure a database](#db-configuration).

### DB configuration

The Docker compose installation already includes the setup of the PostgreSQL database, so you can skip this section.
Any another requires to have a database available to connect to.
The use of [PostgreSQL](https://www.postgresql.org/) is strongly recommended.

#### PostgreSQL

To use PostgreSQL an additional package must be installed:

```shell
pip install psycopg2
```

Then, to connect to PostgreSQL, a dedicated DB must be created.
These are the SQL commands to create the required user and database on PostgreSQL.

```
CREATE USER ics WITH PASSWORD 'ics';
CREATE DATABASE ics;
GRANT ALL PRIVILEGES ON DATABASE ics to ics;
```

These commands can be issued using the `psql` SQL shell (or using pgAdmin, or similar db frontends).
The tables required by ICS are created automatically at the first run.

Then ICS can be launched passing the DB connection string:

```shell
ics-webapp --db_connection_string postgresql://ics:ics@localhost:5432/ics
```

The above connection string is the correct one for a locally running database, change it according to your configuration.

#### SQLite

By default ICS uses SQLite as the DB, yet please note that **the use of SQLite is intended only for a first exploration of ICS and that using PostgreSQL is strongly recommended**.
Using SQLite can result in reduced efficiency and some functionalities may be missing or not properly working.

To use SQLite use the following ``--db_connection_string`` argument to the launch script:

```shell
ics-webapp --db_connection_string sqlite:///ics.sqlite
```

This is the default connection string, it creates the DB file in the current working directory.
Change it to point to the path where you want to store your file.

Again, PostgreSQL is the recommended database.

## <a name="startmain"></a> The main app

Running the docker image automatically starts the main application, which can be accessed with a browser at the ip and 
port defined with the docker launch command or docker compose file.
Installations that do not use docker can run ics by using the ics-webapp script.

Activate the virtual environment:
```shell
conda activate ics
```

When installed using `pip`, the main application can be started with the command:

```shell
ics-webapp
```

When working on source code, it can be launched from the `ics-webapp.py` script:

Linux/Mac:
```shell
PYTHONPATH=. python ics/scripts/ics-webapp.py
```
Windows:
```shell
set PYTHONPATH=. 
python ics/scripts/ics-webapp.py
```

When launched, the app will print the URL at which it is accessible.

```
[30/Mar/2022:15:31:59] ENGINE Bus STARTING
[30/Mar/2022:15:31:59] ENGINE Started monitor thread 'Autoreloader'.
[30/Mar/2022:15:31:59] ENGINE Serving on http://127.0.0.1:8080
[30/Mar/2022:15:31:59] ENGINE Bus STARTED
[30/Mar/2022:15:31:59] ENGINE Started monitor thread 'Session cleanup'.
```

## <a name="login"></a> Login

After the installation, only the `admin` user is defined, with password `adminadmin`.
Change the default password on the first run.

## <a name="configuration"></a> Configuration

A configuration for `ics-webapp` can be saved to a file using the `-s` argument with the filename to use. For example,
this command creates a `default.conf` file that lists all the default values (if any other argument is used in the
command, the value of the argument is saved in the configuration file).

```shell
ics-webapp -s default.conf
```

A configuration file can be used to set the launch arguments, using the `-c` argument:

```shell
ics-webapp -c myinstance.conf
```

Any additional argument passed on the command line overrides the one specified in the configuration file.

## <a name="apps"></a> Additional apps

These apps are clients that connect to the ICS web applications.

If you run ICS from Docker you must install them in a local python environment (``pip install ics-pkg``, note that you don't need to set up the DB for them)

If ICS is not running on the local machine with default port, you must use the ``--host [ip address or name]`` and/or the ``--port [number]`` arguments.
### Command line interface

When the ics-webapp is running, ICS can be also accessed from command line

```
> ics-cli
Welcome, type help to have a list of commands
> login admin
Password: 
'Ok'
>
```

### Twitter stream collector

A command line app, based on [TwiGet](https://github.com/aesuli/twiget), automatically upload to ICS the tweets
collected from filtered stream queries.

```
> ics-twitter-uploader
Logging into http://127.0.0.1:8080/service/userauth/
Username: admin
Password: 
TwiGet 0.1.5

Available commands (type help <command> for details):
create, delete, exit, help, list, refresh, start, stop

Reminder: add -is:retweet to a rule to exclude retweets from results, and to get only original content.
Registered queries:
        no registered queries

[not collecting (0 since last start)]>
```

## <a name="videos"></a> Video tutorials

[This YouTube playlist](https://www.youtube.com/playlist?list=PLde6PofTv7SzplW73XNjiS6zyNyDBfsN9) collects videos showing what you can do with ICS.

## <a name="license"></a> License

This software is licensed under the [3-Clause BSD license](https://opensource.org/licenses/BSD-3-Clause) unless
otherwise noted.
