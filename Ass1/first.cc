/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"

#include "ns3/pcap-file-wrapper.h" //add pcap header file

// Default Network Topology
//
//       10.1.1.0
// n0 -------------- n1
//    point-to-point
//
 
using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("FirstScriptExample");

int
main (int argc, char *argv[])
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);
  
  Time::SetResolution (Time::NS);
  LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);

  NodeContainer nodes;
  nodes.Create (2);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("10Mbps")); //Set the data rate link to 10 mbps
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("5ms")); //Set the delay to 5 mbps

  NetDeviceContainer devices;
  devices = pointToPoint.Install (nodes);

  InternetStackHelper stack;
  stack.Install (nodes); // install internet stack on the node 

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0"); // assign ip address

  Ipv4InterfaceContainer interfaces = address.Assign (devices);

  UdpEchoServerHelper echoServer (9); // port number

  ApplicationContainer serverApps = echoServer.Install (nodes.Get (1)); // install server on node 1
  serverApps.Start (Seconds (1.0));
  serverApps.Stop (Seconds (10.0));

  UdpEchoClientHelper echoClient (interfaces.GetAddress (1), 9);
  echoClient.SetAttribute ("MaxPackets", UintegerValue (5)); // send 5 udp packets
  echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0))); // time interval 1 sec
  echoClient.SetAttribute ("PacketSize", UintegerValue (2048)); // packet size 2048

  ApplicationContainer clientApps = echoClient.Install (nodes.Get (0)); // install client on node 0
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (10.0));

  pointToPoint.EnablePcapAll("logfile"); //EnablePcapAll to have the log

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
