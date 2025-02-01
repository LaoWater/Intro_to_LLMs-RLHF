from google.cloud import bigquery
from bq_setup import get_bigquery_client
from datetime import date
from decimal import Decimal
import json

client = get_bigquery_client()

# Replace this example query with your BigQuery SQL Query
QUERY_TO_RUN = '''
SELECT 
  schedules.gameId,
  games_wide.homeFielder11
FROM 
  `bigquery-public-data`.`baseball`.`schedules` AS schedules
LEFT JOIN `bigquery-public-data`.`baseball`.`games_wide` AS games_wide 
  ON schedules.gameId = games_wide.gameId
  AND games_wide.durationMinutes > 300
WHERE 
  schedules.gameNumber = 1;
'''


def execute_query(query):
    query_job = client.query(query)
    return query_job.result()


def default_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, date):  # Now 'date' is correctly defined
        return obj.isoformat()  # Convert date to ISO format string
    raise TypeError("Type not serializable")


def print_results(results, print_as_array=True):
    column_names = [field.name for field in results.schema]

    if print_as_array:
        dict_results = [dict(zip(column_names, row)) for row in results]
        print(json.dumps(dict_results, indent=4, default=default_serializer))
    else:
        header = " | ".join(column_names)
        print(header)
        print("-" * len(header))

        for row in results:
            formatted_row = " | ".join(str(item) for item in row)
            print(formatted_row)


def main():
    results = execute_query(QUERY_TO_RUN)
    print_results(results, print_as_array=True)


if __name__ == '__main__':
    main()

