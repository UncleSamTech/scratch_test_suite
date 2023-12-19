# Nope Scratch That: A Historical Datasets of Revisions to Scratch Files
>> This is the code base needed to replicate the DataPaper on `Nope Scratch That: A Historical Datasets of Revisions to Scratch Files. This is built off of on this project [Opening the Valve on Pure Data: Collecting the data, creating the csvs, populating the database, and the usage of the database](https://github.com/anishaislam8/visual_code_revisions) It contains the following elements 
- Information Retreival Scripts from World of Code

- Commit Details Information Retreival such as:

    - Commit Messages
    - Author Information
    - Commit Information
    - Content Parents

- Database Construction.

>> Install all the dependencies needed for the project by running `pip3 install requirements.txt`

## Methodology of our Project. 
![Screenshot](/files/msr_flow-1.png)

## Usage of the Database(scratch_revision_db)

### Schema of our Database
![Screenshot](/files/Schema%20(1)-1.png)
### 1. Activate the sqlite command line interface
Our database is embedded in sqlite giving you the flexibility to query our datasets on the go without standing up a seperate server. 
- Load the database using the command : `sqlite3 scratch_revisions_database.db`. This will automatically activate the command line for curating several queries

### 2. SAMPLE QUERIES

>>> Example 1: Load five projects with the highest commit count
```
SELECT Project_Name, Total_Commits
FROM Projects
ORDER BY Total_Commits DESC
LIMIT 5;
```




