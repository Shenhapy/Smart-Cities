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
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/netanim-module.h" // Include the NetAnim module


// Default Network Topology
//
//                           10.1.1.0
// n8   n7   n6   n5   n0 -------------- n1   n2   n3   n4
//  |   |    |    |    |  point-to-point  |    |    |    |
// =====================                   ================
//    LAN2 10.1.3.0                          LAN1 10.1.2.0


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("SecondScriptExample");

int 
main (int argc, char *argv[])
{
  bool verbose = true;
  uint32_t nCsma1 = 3; //Lan1 4 nodes
  uint32_t nCsma2 = 4; //define second lan2 network as it has 5 nodes

  CommandLine cmd (__FILE__);
  cmd.AddValue ("nCsma1", "Number of devices in the first CSMA network", nCsma1); //like cin but for ns3
  cmd.AddValue ("nCsma2", "Number of devices in the second CSMA network", nCsma2);
  cmd.AddValue ("verbose", "Tell echo applications to log if true", verbose);

  cmd.Parse (argc,argv);

  if (verbose)
  {
    LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
    LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);
  }

  nCsma1 = nCsma1 == 0 ? 1 : nCsma1;
  nCsma2 = nCsma2 == 0 ? 1 : nCsma2;

  NodeContainer p2pNodes;
  p2pNodes.Create (2);

  NodeContainer csmaNodes1;
  csmaNodes1.Add (p2pNodes.Get (1)); // Last node of first CSMA network
  csmaNodes1.Create (nCsma1);

  NodeContainer csmaNodes2;
  csmaNodes2.Add (p2pNodes.Get (0)); // First node of second CSMA network
  csmaNodes2.Create (nCsma2);

  //P2P topology with rate of 10Mbps and delay of 2ms.
  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer p2pDevices;
  p2pDevices = pointToPoint.Install (p2pNodes);

  //CSMA topology with data rate of 100Mbps and delay of 50ms. Node n1 from P2P part is also a part of LAN1.
  CsmaHelper csma1;
  csma1.SetChannelAttribute ("DataRate", StringValue ("100Mbps"));
  csma1.SetChannelAttribute ("Delay", StringValue ("50ms"));

  NetDeviceContainer csmaDevices1;
  csmaDevices1 = csma1.Install (csmaNodes1);

  //CSMA topology with data rate of 200Mbps and delay of 20ms. Node n0 from P2P part is also a part of LAN2.
  CsmaHelper csma2;
  csma2.SetChannelAttribute ("DataRate", StringValue ("200Mbps"));
  csma2.SetChannelAttribute ("Delay", StringValue ("20ms"));

  NetDeviceContainer csmaDevices2;
  csmaDevices2 = csma2.Install (csmaNodes2);

  InternetStackHelper stack;
  stack.Install (csmaNodes1);
  stack.Install (csmaNodes2);

  Ipv4AddressHelper address;
  
  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer p2pInterfaces;
  p2pInterfaces = address.Assign (p2pDevices);

  address.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer csmaInterfaces1;
  csmaInterfaces1 = address.Assign (csmaDevices1);

  address.SetBase ("10.1.3.0", "255.255.255.0");
  Ipv4InterfaceContainer csmaInterfaces2;
  csmaInterfaces2 = address.Assign (csmaDevices2);

  //Install mobility on the nodes
  MobilityHelper mobility;
  mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
                                 "MinX", DoubleValue (0.0),
                                 "MinY", DoubleValue (0.0),
                                 "DeltaX", DoubleValue (5.0),
                                 "DeltaY", DoubleValue (10.0),
                                 "GridWidth", UintegerValue (3),
                                 "LayoutType", StringValue ("RowFirst"));

  //Constant position mobility model for this assignment.
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

  mobility.Install (csmaNodes1);
  mobility.Install (csmaNodes2);                              

  UdpEchoServerHelper echoServer (9);

  //The server is installed on the last node of LAN1 (n4).
  ApplicationContainer serverApps = echoServer.Install (csmaNodes1.Get (nCsma1)); // Last node of first CSMA network (n4)
  serverApps.Start (Seconds (1.0));
  serverApps.Stop (Seconds (11.0));

  UdpEchoClientHelper echoClient (csmaInterfaces1.GetAddress (nCsma1), 9); // Last address of second CSMA network
  echoClient.SetAttribute ("MaxPackets", UintegerValue (20));
  echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
  echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

  //The client is installed on the last node of LAN2 (n8).
  ApplicationContainer clientApps = echoClient.Install (csmaNodes2.Get (nCsma2)); // Last node of second CSMA network
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (11.0));

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  pointToPoint.EnablePcapAll ("/home/shenhapy/Downloads/ns-allinone-3.35/ns-3.35/pcap/second");
  csma1.EnablePcap ("/home/shenhapy/Downloads/ns-allinone-3.35/ns-3.35/pcap/second", csmaDevices1.Get (1), true);
  csma2.EnablePcap ("/home/shenhapy/Downloads/ns-allinone-3.35/ns-3.35/pcap/second", csmaDevices2.Get (0), true);

  //Add animation module in order to visualize the simulation.
  // Create a trace file for NetAnim.
  AnimationInterface anim("/home/shenhapy/Downloads/ns-allinone-3.35/animation.xml");

  // Set positions for csmaNodes1 (n1, n2, n3, n4).
  anim.SetConstantPosition(csmaNodes1.Get(0), 10, 10);   // n1
  anim.SetConstantPosition(csmaNodes1.Get(1), 0, 5);  // n2
  anim.SetConstantPosition(csmaNodes1.Get(2), 20, 5);  // n3
  anim.SetConstantPosition(csmaNodes1.Get(3), 10, 0);  // n4

  // Set positions for csmaNodes2 (n0, n5, n6, n7, n8).
  anim.SetConstantPosition(csmaNodes2.Get(0), 10, 17.5);  // n0
  anim.SetConstantPosition(csmaNodes2.Get(1), 0, 20);  // n5
  anim.SetConstantPosition(csmaNodes2.Get(2), 20, 15);  // n6
  anim.SetConstantPosition(csmaNodes2.Get(3), 20, 20); // n7
  anim.SetConstantPosition(csmaNodes2.Get(4), 10, 25); // n8

  Simulator::Run ();
  Simulator::Destroy ();

  return 0;
}
