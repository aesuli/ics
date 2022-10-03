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

### Installation: using pip (recommended)

The suggested way to quickly set up the python enviroment is to use
the [Anaconda/Miniconda distribution](https://www.anaconda.com/products/distribution) and the `conda` package manager to
create the virtual enviroment.

ICS is published as a [`pip` package](https://pypi.org/project/ics-pkg).

```
> conda create -n ics python
> conda activate ics
> pip install ics-pkg
```

### Installation: from source

Download source code from [GitHub repo](https://github.com/aesuli/ics).

```
> cd [directory with ICS code]
> conda create -n ics --file requirements.txt
> conda activate ics
```

Note: twiget is not listed as a requirement, as it is needed only by the twitter uploader script (`pip install twiget`).

### DB configuration

ICS requires a database to store its data.

By default, ICS assumes the use of a database named 'ics' by a user named 'ics' (with password 'ics').

ICS is tested to work with [PostgreSQL](https://www.postgresql.org/). These are the SQL commands to create the required
user and database on PostgreSQL.

```
    CREATE USER ics WITH PASSWORD 'ics';
    CREATE DATABASE ics;
    GRANT ALL PRIVILEGES ON DATABASE ics to ics;
```

These command can be issued using the `psql` SQL shell (or using pgAdmin, or similar db frontends).

The tables required by ICS are created automatically at the first run.

## <a name="startmain"></a> Starting the main app

Activate the virtual environment:

```
> conda activate ics
```

When installed using `pip`, the main application can be started with the command:

```
> ics-webapp
```

When working on source code, it can be launched from the `ics-webapp.py` script:

```
Linux/Mac:
>PYTHONPATH=. python ics/scripts/ics-webapp.py

Windows:
>set PYTHONPATH=. 
>python ics/scripts/ics-webapp.py
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

## <a name="configuration"></a> Configuration

A configuration for `ics-start` can be saved to a file using the `-s` argument with the filename to use. For example,
this command creates a `default.conf` file that lists all the default values (if any other argument is used in the
command, the value of the argument is saved in the configuration file).

```
> ics-start -s default.conf
```

A configuration file can be used to set the launch arguments, using the `-c` argument:

```
> ics-start -c myinstance.conf
```

Any additional argument passed on the command line overrides the one specified in the configuration file.

## <a name="apps"></a> Additional apps

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
