#include <fstream>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include <unistd.h>
#include <thread>

#include "ns3/point-to-point-layout-module.h"
#include "ns3/traffic-control-module.h"

#include <iostream>
#include <iomanip>
#include <map>


#include <vector>
#include <cmath>
#include <algorithm>

#include "ns3/ns3-ai-module.h"
#include "ns3/log.h"


using namespace std;
using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("FifthScriptExample");

//////////////////////////NS3-AI////////////////////////////////////////////////

struct Env
{
    int client_num;
    bool clientUpdateFlag;
    bool isRoundFinished;
}Packed;

struct Act
{
    float client_accuracy;
    float server_accuracy;
}Packed;

class FL : public Ns3AIRL<Env, Act>
{
public:
    FL(uint16_t id);
    float FL_ReceiveData();
    double GetClientReturnAccuracy(int clientId);
    void SetClientUpdateFlag(bool flag);
};

FL::FL(uint16_t id) : Ns3AIRL<Env, Act>(id) {
    SetCond(2, 0);      //< Set the operation lock (even for ns-3 and odd for python).
}

float FL::FL_ReceiveData() {
    auto env = EnvSetterCond();
    env->isRoundFinished = true;
    SetCompleted();

    auto act = ActionGetterCond();
    double score = act->server_accuracy;
    GetCompleted();

    return score;
}

double FL::GetClientReturnAccuracy(int givenClientId)
{
  auto env = EnvSetterCond();
  env->clientUpdateFlag = true;
  env->client_num = givenClientId;
  SetCompleted();

  auto act = ActionGetterCond();
  double ret = act->client_accuracy;
  GetCompleted();
  return ret;
}

void FL::SetClientUpdateFlag(bool flag)
{
  auto env = EnvSetterCond();
  env->clientUpdateFlag = flag;
  SetCompleted();
}


///////////////////////////////////////////////////////////////////////////////

// GLOBAL VALUES
bool tracing;
uint8_t m_clients;
uint8_t m_server;

class MyApp : public Application
{
public:

  MyApp ();
  virtual ~MyApp();

  void Setup (uint32_t Id, Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate, Ipv4Address ipAddress, Ptr<FL> fl);
  void Setup(uint32_t Id, vector<Ptr<MyApp>> clientAppList, uint32_t numOfClients, Ptr<Socket> socket, Ipv4Address ipAddress, Ptr<FL> fl);
  
  uint32_t m_Id;
  bool     is_sent_all;
  bool     waiting;

private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);

  void ScheduleTx (void);
  void SendPacket (void);
  void onReceive (Ptr<Socket> socket);
  void onAccept (Ptr<Socket> s, const Address& from);

  Ptr<Socket>     m_socket;
  Address         m_peer;
  uint32_t        m_packetSize;
  uint32_t        m_nPackets;
  DataRate        m_dataRate;
  EventId         m_sendEvent;
  bool            m_running;
  uint32_t        m_packetsSent;
  bool            m_isServer;
  Ipv4Address     m_ipAddress;
  

  vector<Ptr<MyApp>> m_clientAppList;
  uint32_t m_numOfClients;
  Ptr<FL> m_fl;
};

MyApp::MyApp ()
  : m_socket (0),
    m_peer (),
    m_packetSize (0),
    m_nPackets (0),
    m_dataRate (0),
    m_sendEvent (),
    m_running (false),
    m_packetsSent (0),
    m_isServer (false)
{
}

MyApp::~MyApp()
{
  m_socket = 0;
}

void
MyApp::Setup (uint32_t Id,Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate, Ipv4Address ipAddress, Ptr<FL> fl)
{
  m_socket = socket;
  m_peer = address;
  m_packetSize = packetSize;
  m_nPackets = nPackets;
  m_dataRate = dataRate;
  m_Id = Id;
  m_ipAddress = ipAddress;

  is_sent_all = false;
  m_fl = fl;
}

void
MyApp::Setup(uint32_t Id, vector<Ptr<MyApp>> clientAppList, uint32_t numOfClients, Ptr<Socket> socket, Ipv4Address ipAddress, Ptr<FL> fl)
{
    m_socket = socket;
    m_isServer = true;
    m_Id = Id;

    m_clientAppList = clientAppList;
    m_numOfClients = numOfClients;
    m_ipAddress = ipAddress;

    waiting = true;
    m_fl = fl;
}


void
MyApp::StartApplication (void)
{
  m_running = true;
  if (!m_isServer) {
      m_packetsSent = 0;
      m_socket->Connect (m_peer);
      NS_LOG_UNCOND("Client: "<<m_ipAddress<<" m_peer:"<<m_peer);
      
      SendPacket ();
  } else {

      m_socket->Listen();

      m_socket->SetAcceptCallback (MakeNullCallback<bool, Ptr<Socket>, const Address &> (), MakeCallback (&MyApp::onAccept, this));

      NS_LOG_UNCOND("Server: "<<m_ipAddress);
  }
}

void
MyApp::StopApplication (void)
{
  m_running = false;

  if (m_sendEvent.IsRunning ())
    {
      Simulator::Cancel (m_sendEvent);
    }

  if (m_socket)
    {
      m_socket->Close ();
    }
}

void
MyApp::SendPacket (void)
{
  
  if (++m_packetsSent < m_nPackets)
    {
      Ptr<Packet> packet = Create<Packet> (m_packetSize);
      int result = m_socket->Send (packet);

      if (result < 0)
      {
        NS_LOG_INFO("Error is occured while sending. Error no: " << m_socket->GetErrno());
      }
      else
      {
        // NS_LOG_INFO("Sent!");
      }
      ScheduleTx ();
    }
  if (m_packetsSent == m_nPackets && is_sent_all==false)
    {
      NS_LOG_UNCOND("All Packets Are Sent by Client_"<<m_Id<<"!!!");
      //m_fl->FL_Trigger(m_Id);
      is_sent_all = true;
    }
    
}

void
MyApp::ScheduleTx (void)
{
  if (m_running)
    {
      Time tNext (Seconds (m_packetSize * 8 / static_cast<double> (m_dataRate.GetBitRate ())));
      m_sendEvent = Simulator::Schedule (tNext, &MyApp::SendPacket, this);
    }
}

void MyApp::onReceive(Ptr<Socket> socket) {
  Ptr<Packet> packet;
  while ((packet = socket->Recv())) {
    if (packet->GetSize () == 0) { //EOF 
      break;
    }
  }
  
  bool flag = true;
  for (uint8_t i=0;i<m_numOfClients-1;i++) {
    flag = m_clientAppList[i]->is_sent_all;
    if (flag == false) {
      break;
    }
  }

  if (flag && waiting) {

    for (uint8_t i=0;i<m_numOfClients-1;i++) {
      if (m_clientAppList[i]->is_sent_all) {
        //m_fl->GetClientReturnAccuracy(m_clientAppList[i]->m_Id);
        double c_accuracy = m_fl->GetClientReturnAccuracy(m_clientAppList[i]->m_Id);
        NS_LOG_UNCOND("Client_"<<m_clientAppList[i]->m_Id<<" Accuracy: "<<c_accuracy);
        m_fl->SetClientUpdateFlag(false); 
      }
    }
    
    NS_LOG_UNCOND("All Packets are Received!!!");

    float s_accuracy = m_fl->FL_ReceiveData();
    NS_LOG_UNCOND("\n");
    NS_LOG_UNCOND("Overall Accuracy Result: "<< s_accuracy);

    waiting = false;
  }

}

void MyApp::onAccept(Ptr<Socket> s, const Address& from) {
    //THIS IS WHERE THE CALLBACK GETS SET
    s->SetRecvCallback(MakeCallback(&MyApp::onReceive, this));
}


int
RunSimulation ()
{
  //*Set Seed
  SeedManager::SetRun(441);
  RngSeedManager::SetSeed(441);
  
  Ptr<DefaultSimulatorImpl> s = CreateObject<DefaultSimulatorImpl> ();
  Simulator::SetImplementation(s);

  std::string bitRate = "5Mbps";//"5Mbps";
  std::string delay = "1ms";//"1ms";
  uint8_t npacket = 5;

  int memblock_key = 2333;

  // Create AI part
  Ptr<FL> flPtr = Create<FL>(memblock_key);

  // Create the point-to-point link helpers
  PointToPointHelper pointToPointRouter, p2p_all;
  pointToPointRouter.SetDeviceAttribute("DataRate", StringValue(bitRate));
  pointToPointRouter.SetChannelAttribute("Delay", StringValue(delay));
  //pointToPointRouter.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("1p"));
  PointToPointHelper pointToPointLeaf;
  pointToPointLeaf.SetDeviceAttribute("DataRate", StringValue(bitRate));
  pointToPointLeaf.SetChannelAttribute("Delay", StringValue(delay));

  PointToPointDumbbellHelper dumbbellHelper(m_clients, pointToPointLeaf,
                                            m_server, pointToPointLeaf,
                                            pointToPointRouter);

  NS_LOG_UNCOND("Create nodes.");
  NodeContainer nodes = NodeContainer(dumbbellHelper.GetLeft(), dumbbellHelper.GetRight());
  NetDeviceContainer devices = p2p_all.Install(nodes);
  
  // Install Stack
  InternetStackHelper stack;
  dumbbellHelper.InstallStack(stack);

  NS_LOG_UNCOND("Assign IP Addresses.");
  // Assign IP Addresses
  dumbbellHelper.AssignIpv4Addresses(Ipv4AddressHelper("10.1.0.0", "255.255.255.0"),
                                     Ipv4AddressHelper("10.2.1.0", "255.255.255.0"),
                                     Ipv4AddressHelper("10.3.1.0", "255.255.255.0"));
  
  NS_LOG_UNCOND("Create sockets.");

  //Receiver socket on n1
  //TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
  Ptr<Socket> server = Socket::CreateSocket(dumbbellHelper.GetRight(0), TcpSocketFactory::GetTypeId ());
  InetSocketAddress local = InetSocketAddress(dumbbellHelper.GetRightIpv4Address(0), 4477);
  server->Bind(local);

  //Sender sockets
  vector<Ptr<Socket>> clientSocketList;
  vector<Ptr<MyApp>> flClientAppList;
  for (uint32_t i = 0; i < dumbbellHelper.LeftCount(); i++)
  {
    Ptr<Socket> client = Socket::CreateSocket(dumbbellHelper.GetLeft(i), TcpSocketFactory::GetTypeId ());
    clientSocketList.push_back(client);

    Ptr<MyApp> flApp = CreateObject<MyApp>();
    flApp->Setup(i+1,client, local, 63507, npacket, DataRate(bitRate), dumbbellHelper.GetLeftIpv4Address(i), flPtr);


    flClientAppList.push_back(flApp);
  }
  
  // Client app
  for (uint32_t i = 0; i < dumbbellHelper.LeftCount() - 1; i++)
  {
    dumbbellHelper.GetLeft(i)->AddApplication(flClientAppList[i]);
    flClientAppList[i]->SetStartTime(Seconds(2));
    
  }

  // Server app:
  Ptr<MyApp> serverApp = CreateObject<MyApp>();
  dumbbellHelper.GetRight(0)->AddApplication(serverApp);
  serverApp->Setup(0,flClientAppList, dumbbellHelper.LeftCount(),server, dumbbellHelper.GetRightIpv4Address(0), flPtr);
  serverApp->SetStartTime(Seconds(1));
  //serverApp->TraceConnectWithoutContext("RoundCheckerTrace", MakeBoundCallback(&RoundTrace, serverApp));
  //server->SetRecvCallback(MakeBoundCallback(&ReceivePacket, pktSentStream, serverApp));

  //Simulator::Stop (Seconds (20));
  Ipv4GlobalRoutingHelper::PopulateRoutingTables();

  NS_LOG_UNCOND("Run Simulation.");
  Simulator::Run ();
  Simulator::Destroy ();

  return 0;
}

int 
main (int argc, char *argv[])
{
  m_clients = 20;//20; Left Leaf
  m_server = 1; // Right Leaf
  tracing = false;
  
  CommandLine cmd (__FILE__);
  //cmd.AddValue ("m_server", "Number of Server nodes/devices", m_server);
  cmd.AddValue ("client_num", "Number of Clients per Server", m_clients);
  //cmd.AddValue ("tracing", "Enable pcap tracing", tracing);
  cmd.Parse (argc, argv);


  m_clients = m_clients+1;
  RunSimulation();

  return 0;
}


