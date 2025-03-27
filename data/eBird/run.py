import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from config import config
import filter_data
import combine_summarize

# The data is first filtered down to the bounding box and time frame specified in the config.
print("Running data filtering...")
filter_data.main() 

# Processing is done in R using the auk package which is regularly updated and made specifically for use with eBird data.
print("Running checklist aggregation...")
ro.r('source("/data/cher/EcoBound/data/eBird/process_eBird.R")')
process_ebird = ro.r['process_ebird']
process_ebird(config.beginning_date, config.ending_date, config.output_path)

# Combine, summarize and clean up
print("Combining, summarization and cleaning...")
combine_summarize.main() 
