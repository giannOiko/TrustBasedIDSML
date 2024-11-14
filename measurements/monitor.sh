#!/bin/bash

parent_pid=$PPID

# Use ps to get details about the parent process
parent_process=$(ps -o comm= -p "$parent_pid")

while true; do
  # Get the system-wide CPU and memory usage
  system_cpu=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')  # User + System CPU usage
  system_mem=$(free -m | awk 'NR==2{printf "%.2f", $3*100/$2 }')  # Memory usage in percentage
  echo "%System CPU: $system_cpu %  %System MEM: $system_mem %  called by: $parent_process" >> ps.log;  done;