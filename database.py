from flask import Flask, jsonify
from flask import request
import mysql.connector

app = Flask(__name__)

# MySQL Configuration
db_config = {
    'user': 'root',
    'password': 'redeem419',
    'host': 'localhost',
    'port': '3307',
    'database': 'nnuro_dev',
    'raise_on_warnings': True
}
@app.route('/api/db-connect/', methods=['GET'])
def get_data():
    # Connect to MySQL
    cnx = mysql.connector.connect(**db_config)
    cursor = cnx.cursor()

    # Execute a sample query
    query = 'SELECT * FROM companies'
    cursor.execute(query)

    # Fetch all records from the result
    data = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    cnx.close()

    # Convert data to JSON format
    result = []
    for row in data:
        result.append({
            'column1': row[0],
            'column2': row[1],
            # Add more columns as needed
        })

    # Return JSON response
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
