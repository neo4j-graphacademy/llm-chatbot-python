library("tidyverse")
library("dplyr")

load_data <- function(files) {
  data <- lapply(files, read_csv)
  names(data) <- gsub(pattern = ".csv", replacement = "", basename(files))
  return(data)
}

filter_data <- function(data) {
  # 0. limit number of exposure relations (due to aura db no. of relations limit)
  #    use supplemented, [algae, fish], sort desc by TU and take the top 10 million data points
  data$exposure <- data$exposure %>%
    filter(tox_stat == "supplemented", species == "algae") %>%
    arrange(desc(TU)) %>%
    head(200000)
  # 1. find measured substances
  measured_substances <- data$exposure %>% select(DTXSID) %>% unique()
  # 2. intersect measured substances with hazard information, select supplemented tox_stat
  data$hazard <- data$hazard %>%
    filter(tox_stat == "supplemented", species == "algae") %>%
    right_join(measured_substances, by = join_by("DTXSID"))
  substances_with_hazard_information <- data$hazard %>% select(DTXSID) %>% unique()
  # 3. interesect substances with hazard information with exposure
  data$exposure <- data$exposure %>%
    right_join(substances_with_hazard_information, by = join_by("DTXSID"))
  # 4. interesect substances with hazard information with substances
  data$substances <- data$substances %>%
    select(-Norman_SusDat_ID) %>%
    unique() %>%
    right_join(substances_with_hazard_information, by = join_by("DTXSID"))
  # 5. extract remaining drivers
  data$drivers <- data$drivers %>%
    filter(tox_stat == "supplemented", species == "algae") %>%
    right_join(substances_with_hazard_information, by = join_by("DTXSID"))
  # 6. extract remaining sites, also for summarized data
  relevant_sites <- data$exposure %>%
    select(station_name_n) %>%
    unique()
  data$sites <- data$sites %>%
    right_join(relevant_sites, by = join_by("station_name_n"))
  data$summarized <- data$summarized %>%
    filter(tox_stat == "supplemented", species == "algae") %>%
    right_join(relevant_sites, by = join_by("station_name_n"))

  return(data)
}

qa_data <- function(data) {
  # Perform quality assurance on the filtered data
  substances <- data$substances %>% pull(DTXSID)
  sites <- data$sites %>% pull(station_name_n)
  substances_hazard <- data$hazard %>% pull(DTXSID) %>% unique()
  substances_exposure <- data$exposure %>% pull(DTXSID) %>% unique()
  sites_exposure <- data$exposure %>%
    pull(station_name_n) %>%
    unique()
  return(all(substances_hazard %in% substances) &
           all(sites_exposure %in% sites) &
           all(substances_exposure %in% substances))
}

write_data <- function(data, directory) {
  lapply(names(data), function(name) {
    out_file <- file.path("examples/ChEOS", paste0(name, "_filtered.csv"))
    out_path <- file.path(directory, basename(out_file))
    write_csv(data[[name]], file = out_path)
    system2(command = "ln", args = c("-s", out_path, out_file))
  })
}

main <- function() {
  log_file <- file("examples/ChEOS/subset_showcase.log", open = "a")

  files <- list.files(path = "examples/ChEOS", pattern = ".csv", full.names = TRUE)
  files <- files[grep("_filtered", files, invert = TRUE)]

  data <- load_data(files = files)
  sink(log_file)
  print(paste("Data loaded from:", files))
  sink()

  data_filtered <- filter_data(data = data)

  data_consistent <- qa_data(data = data_filtered)
  sink(log_file)
  print(paste("Data consistent? -", data_consistent))
  sink()

  rm(data)

  data_dir <- dirname(Sys.readlink(files[1]))
  write_data(data = data_filtered, directory = data_dir)
  sink(log_file)
  print(paste("Filtered data written to", data_dir, "with postfix _filtered in the filename."))
  print(paste("Symbolic links created that point to these files."))
  sink()
}

main()

