import mysql.connector

conn = mysql.connector.connect(host = 'localhost', user = 'root', password = '12345', database = 'pythondb')
mycursor = conn.cursor()

sql = "insert into student (name, branch, id) values (%s, %s, %s)"
#val = ("John", "CSE", 56)

#if user want to create multiple value then you can not create list
val = [
    ("John", "CSE", 56),
    ("Peter", "ECE", 57),
    ("Amy", "IT", 58)
]
mycursor.executemany(sql, val)
conn.commit()
print(mycursor.rowcount, "record inserted")