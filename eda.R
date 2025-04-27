# Step 1: EDA
# Visualize time series data
# general stats on distribution of limit balance, sex, ed, marriage, age
# 
# Step 2: Variable Treatment
# Sex, Ed, Marriage >>> One hot encoded
# Ed combine 4+
# 
# Step 3: Feature Engineering
# TS Debt to limit ratio or Average debt to limit ratio (Credit util)
# Average Bill
# Average Payment
# Late Payment Count
# Debt Difference
# 
# Step 4: Feature Scaling 
# normalize all numerical data
# 
# Step 5: Feature Selection
#
# Step 6: Model Refinement
# LOOCV
# Pruning

library(tidyverse)

# resetdata <- function() {
#   data <- read.csv("UCI_Dataset.csv")
#   data <- data[, -1]
#   colnames(data)[colnames(data) == "default.payment.next.month"] <- "DEFAULT"
#   assign("cc", data, envir = .GlobalEnv)
# }

cc <- read.csv("UCI_Dataset.csv")
cc <- cc[, -1]
colnames(cc)[colnames(cc) == "default.payment.next.month"] <- "DEFAULT"

cc$MARRIAGE[cc$MARRIAGE == 3] <- 0

# cc <- cc %>%
#   rename(LIM = LIMIT_BAL, REPAY1 = PAY_0, REPAY2 = PAY_2, REPAY3 = PAY_3, REPAY4 = PAY_4, REPAY5 = PAY_5, REPAY6 = PAY_6) %>%
#   rename_with(.fn = function(x) gsub("BILL_AMT", "BILL", x), .cols = starts_with("BILL_AMT")) %>%
#   rename_with(.fn = function(x) gsub("PAY_AMT", "PAY", x), .cols = starts_with("PAY_AMT"))
# 
# bill_ot <- colMeans(cc[, paste0("BILL", 1:6)])
# pay_ot <- colMeans(cc[, paste0("REPAY", 1:6)])
# pay_amt_ot <- colMeans(cc[, paste0("PAY", 1:6)])
# 
# plot(pay_ot)
# plot(bill_ot)
# plot(pay_amt_ot)
# 
# for (col_name in names(cc)) {
#   hist(cc[[col_name]], 
#        main = paste("Histogram of", col_name), # Title of the histogram
#        xlab = col_name, # Label for the x-axis
#        col = "lightblue", # Color of the bars (optional)
#        border = "black") # Color of the bar borders (optional)
# }
# 
# for (col_name in names(cc[,12:23])) {
#   boxplot(cc[[col_name]], 
#        main = paste("Boxplot of", col_name), # Title of the histogram
#        xlab = col_name, # Label for the x-axis
#        col = "lightblue", # Color of the bars (optional)
#        border = "black") # Color of the bar borders (optional)
# }

# Feature Engineering

# Credit utilization
# cc <- cc %>%
#   mutate(
#     UTIL1 = BILL_AMT1 / LIMIT_BAL,
#     UTIL2 = BILL_AMT2 / LIMIT_BAL,
#     UTIL3 = BILL_AMT3 / LIMIT_BAL,
#     UTIL4 = BILL_AMT4 / LIMIT_BAL,
#     UTIL5 = BILL_AMT5 / LIMIT_BAL,
#     UTIL6 = BILL_AMT6 / LIMIT_BAL
#   )

#Percent Credit Utilization
# cc <- cc %>%
#   mutate(UTIL = rowMeans(select(., starts_with("BILL"))) / LIM)
# 
# cc_util <- cc %>% 
#   filter(if_all(starts_with("UTIL"), ~ between(., 0, 1))) 
# 
# util_ot <- cc_util %>%
#   summarize(across(starts_with("UTIL"), mean)) %>%
#   as.numeric()
# 
# heatmap(cor(cc))
# 
# corrplot(cor(cc), method = "square")

#
# EDA Summary
# Debt balances tend to decrease over time
# Credit utilization over time seems to match debt balances plot
# The amount paid per month seems to be random
# The bill per month seems to be highly consistent
# Repayment behavior seems to be somewhat consistent also
#

library(ggplot2)
cc$default_factor <- factor(cc$DEFAULT, levels = c(0, 1), labels = c("Non-Default", "Default"))

# Default vs Non Default
ggplot(cc, aes(x = default_factor)) +
  geom_bar(fill = c("darkgray", "darkorange2"), color = "black") + # You can customize the colors
  labs(
    title = "Default Imbalance",
    x = "Default Status",
    y = "Number of Individuals"
  ) +
  theme_classic() +
  theme(plot.title = element_text(size = 20, hjust = 0.5),       # Title size
        axis.title.x = element_text(size = 16),                 # X-axis label size
        axis.title.y = element_text(size = 16),                 # Y-axis label size
        axis.text.x = element_text(size = 12),                  # X-axis tick label size
        axis.text.y = element_text(size = 12),                  # Y-axis tick label size
  )


# Age Distribution
ggplot(cc, aes(x = AGE)) +
  geom_histogram(stat = "bin", fill = "darkgray", color = "black", binwidth = 5) +
  labs(
    title = "Age Distribution",
    x = "Age",
    y = "Number of Individuals"
  ) +
  theme_classic() +
  theme(plot.title = element_text(size = 20, hjust = 0.5),       # Title size
        axis.title.x = element_text(size = 16),                 # X-axis label size
        axis.title.y = element_text(size = 16),                 # Y-axis label size
        axis.text.x = element_text(size = 12),                  # X-axis tick label size
        axis.text.y = element_text(size = 12),                  # Y-axis tick label size
        )


# Marriage Distribution
ggplot(cc, aes(x = factor(MARRIAGE))) +
  geom_bar(fill = "darkgray", color = "black") + # You can customize the colors
  labs(
    title = "Marriage Distribution",
    x = "Relationship Status",
    y = "Number of Individuals"
  ) +
  scale_x_discrete(labels = c("Other", "Single", "Married")) +
  theme_classic() +
  theme(plot.title = element_text(size = 20, hjust = 0.5),       # Title size
        axis.title.x = element_text(size = 16),                 # X-axis label size
        axis.title.y = element_text(size = 16),                 # Y-axis label size
        axis.text.x = element_text(size = 12),                  # X-axis tick label size
        axis.text.y = element_text(size = 12),                  # Y-axis tick label size
  ) 

# Sex Distribution
ggplot(cc, aes(x = factor(SEX))) +
  geom_bar(fill = "darkgray", color = "black") + # You can customize the colors
  labs(
    title = "Sex Distribution",
    x = "Sex",
    y = "Number of Individuals"
  ) +
  scale_x_discrete(labels = c("Male", "Female")) +
  theme_classic() +
  theme(plot.title = element_text(size = 20, hjust = 0.5),       # Title size
        axis.title.x = element_text(size = 16),                 # X-axis label size
        axis.title.y = element_text(size = 16),                 # Y-axis label size
        axis.text.x = element_text(size = 12),                  # X-axis tick label size
        axis.text.y = element_text(size = 12),                  # Y-axis tick label size
)




