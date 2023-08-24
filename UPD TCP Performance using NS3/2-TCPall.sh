#!/bin/bash

# Define arrays to store the output data
txPacketsArray=()
txBytesArray=()
txOfferedArray=()
rxPacketsArray=()
rxBytesArray=()
throughputArray=()
avgJitterArray=()
avgDelayArray=()
lostPacketsArray=()
lastPacketSentArray=()
lastPacketReceivedArray=()
totalBytesReceivedArray=()
pc=0

for x in $(seq 1 11)
do
    pc=$((pc + 15000000))
    output=$(./waf --run "scratch/1-tcp.cc --SendSize=$pc")
    
    # Extract relevant values from the output and append them to the arrays
    txPacketsArray+=($(echo "$output" | awk '/Tx Packets:/ {print $3}'))
    txBytesArray+=($(echo "$output" | awk '/Tx Bytes:/ {print $3}'))
    txOfferedArray+=($(echo "$output" | awk '/TxOffered:/ {print $2}'))
    rxPacketsArray+=($(echo "$output" | awk '/Rx Packets:/ {print $3}'))
    rxBytesArray+=($(echo "$output" | awk '/Rx Bytes:/ {print $3}'))
    throughputArray+=($(echo "$output" | awk '/Throughput:/ {print $2}'))
    avgJitterArray+=($(echo "$output" | awk '/Average Jitter:/ {print $3}'))
    avgDelayArray+=($(echo "$output" | awk '/Average Delay:/ {print $3}'))
    lostPacketsArray+=($(echo "$output" | awk '/Lost packets:/ {print $3}'))
    lastPacketSentArray+=($(echo "$output" | awk '/Last packet sent at:/ {print $6}'))
    lastPacketReceivedArray+=($(echo "$output" | awk '/Last packet received at:/ {print $6}'))
    totalBytesReceivedArray+=($(echo "$output" | awk '/Total Bytes Received:/ {print $4}'))
done

# Display the arrays side by side
echo "Tx Packets: ${txPacketsArray[*]}"
echo "Tx Bytes: ${txBytesArray[*]}"
echo "TxOffered: ${txOfferedArray[*]}"
echo "Rx Packets: ${rxPacketsArray[*]}"
echo "Rx Bytes: ${rxBytesArray[*]}"
echo "Throughput: ${throughputArray[*]}"
echo "Average Jitter: ${avgJitterArray[*]}"
echo "Average Delay: ${avgDelayArray[*]}"
echo "Lost packets: ${lostPacketsArray[*]}"
echo "Last packet sent at: ${lastPacketSentArray[*]}"
echo "Last packet received at: ${lastPacketReceivedArray[*]}"
echo "Total Bytes Received: ${totalBytesReceivedArray[*]}"

