# process_ebird.R

# Install and load required packages if they're not already installed
required_packages <- c("auk", "dplyr", "ggplot2", "gridExtra", "lubridate", "readr", "tidyverse", "progress")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

process_ebird <- function(beginning_date, ending_date, output_path) {
  # Now that the packages are loaded, proceed with the rest of the function

  # Create date range.
  beginning_date <- as.Date(beginning_date)
  ending_date   <- as.Date(ending_date)
  date_seq <- seq(beginning_date, ending_date, by = "month")
  date_pairs <- data.frame(year = year(date_seq), month = month(date_seq))

  pb <- progress_bar$new(
  format = "Processing [:bar] :percent (:current/:total) :elapsed",
  total = nrow(date_pairs),
  width = 60
  )
  
  # Original data
  f_sed <- paste0(output_path, "stl_ebird_sampling_data.txt")
  f_ebd <- paste0(output_path, "stl_ebird_observations_data.txt")

  checklists <- read_sampling(f_sed)
  observations <- read_ebd(f_ebd)

  # Only keep complete checklists
  checklists <- checklists %>% filter(all_species_reported == TRUE)
  
  # Loop through all year-month pairs
  for (i in seq_len(nrow(date_pairs))) {
    year_i <- date_pairs$year[i]
    month_i <- date_pairs$month[i]

    pb$tick()

    # Filter data for the current year and month
    checklists_filtered <- checklists |> 
      filter(year(observation_date) == year_i, month(observation_date) == month_i)
    
    observations_filtered <- observations |> 
      filter(year(observation_date) == year_i, month(observation_date) == month_i)
    
    # Ensure observations match filtered checklists
    observations_filtered <- semi_join(observations_filtered, checklists_filtered, by = "checklist_id")
    
    if (nrow(observations_filtered) == 0) next  # Skip if no observations for this month
    
    zf <- auk_zerofill(observations_filtered, checklists_filtered, collapse = TRUE)
    
    # Function to convert observation time to hours since midnight
    time_to_decimal <- function(x) {
      x <- hms(x, quiet = TRUE)
      hour(x) + minute(x) / 60 + second(x) / 3600
    }
    
    # Process data
    zf_filtered <- zf |> 
      mutate(
        effort_distance_km = if_else(protocol_type == "Stationary", 0, effort_distance_km),
        effort_hours = duration_minutes / 60,
        effort_speed_kmph = effort_distance_km / effort_hours,
        hours_of_day = time_to_decimal(time_observations_started),
        year = year(observation_date),
        day_of_year = yday(observation_date)
      ) |> 
      filter(protocol_type %in% c("Stationary", "Traveling"),
              effort_hours <= 6,
              effort_distance_km <= 10,
              effort_speed_kmph <= 100,
              number_observers <= 10) |> 
      select(checklist_id, observer_id, observation_date, hours_of_day, latitude, longitude, 
              observation_count, species_observed, scientific_name, 
              effort_hours, effort_distance_km, effort_speed_kmph, 
              number_observers)
    
    # Convert species_observed to numeric (TRUE -> 1, FALSE -> 0) and pivot wider
    zf_wide <- zf_filtered %>%
      mutate(species_observed = as.integer(species_observed)) %>%  
      pivot_wider(names_from = scientific_name, values_from = species_observed, values_fill = list(species_observed = 0))
    
    # Define output file name
    output_file <- sprintf("%seBird_%d_%02d.csv", output_path, year_i, month_i)
    
    # Save to CSV
    write_csv(zf_wide, output_file, na = "")
    
    print(paste("Saved:", output_file))
  }
}