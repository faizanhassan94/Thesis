# A framework to correlate resource behavior with the success of process instances: a case study in a P2P process
Repository containing Python scripts to extract resource behavior and identify its success rate.

## Important!
The original case study is not publicly available. The analysis has been performed on a public log.


## Installation

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using command : "pip install -r requirements.txt."

## Usage

1. Start the development server by running the command: "python manage.py runserver."
2. The development server will start and would be accessible at http://127.0.0.1:8000/.
3. On the home page, you will find an option to upload an event log file.
4. Select a KPI from the dropdown list of available KPIs.
5. Click the "Submit" button to initiate the analysis.
6. Wait for the framework to process the uploaded event log and calculate the success rates.
7. Once the analysis is complete, the results will be displayed on the web page.

Please make sure that the event log files you upload are in the XES format. If you have event logs in a different format, you may need to convert them to XES before using this web application.
