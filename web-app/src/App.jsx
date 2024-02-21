import React from 'react'
import { Button, Grid, Link, TextField, Switch, FormControlLabel } from '@mui/material'
import './App.css'
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css'
import { MainContainer, ChatContainer, MessageList, Message, MessageInput, TypingIndicator } from '@chatscope/chat-ui-kit-react'
import 'bootstrap/dist/css/bootstrap.min.css'
import { Container, Row, Col } from 'react-bootstrap'

function App() {
  // REACT COMPONENTS TO HANDLE TRAINING
  const lossNodeName = "onnx::loss::8";

    const logIntervalMs = 1000;
    const waitAfterLoggingMs = 500;
    let lastLogTime = 0;

    let messagesQueue = [];

    // React components
    const [maxNumTrainSamples, setMaxNumTrainSamples] = React.useState(6400);
    const [maxNumTestSamples, setMaxNumTestSamples] = React.useState(1280);

    const [batchSize, setBatchSize] = React.useState(64);
    const [numEpochs, setNumEpochs] = React.useState(5);

    const [trainingLosses, setTrainingLosses] = React.useState([]);
    const [testAccuracies, setTestAccuracies] = React.useState([]);

    const [isTraining, setIsTraining] = React.useState(false);

    const [moreInfoIsCollapsed, setMoreInfoIsCollapsed] = React.useState(true);

    // logging React components
    const [enableLiveLogging, setEnableLiveLogging] = React.useState(false);

    const [statusMessage, setStatusMessage] = React.useState("");
    const [errorMessage, setErrorMessage] = React.useState("");
    const [loggingMessages, setLoggingMessages] = React.useState([]);

    // logging helper functions

    function toggleMoreInfoIsCollapsed() {
        setMoreInfoIsCollapsed(!moreInfoIsCollapsed);
    }

    function showStatusMessage(message) {
        console.log(message);
        setStatusMessage(message);
    }

    function showErrorMessage(message) {
        console.log(message);
        setErrorMessage(message);
    }

    function addMessages(messagesToAdd) {
        setLoggingMessages(loggingMessages => [...loggingMessages, ...messagesToAdd]);
    }

    function addMessageToQueue(message) {
        messagesQueue.push(message);
    }

    function clearOutputs() {
        setTrainingLosses([]);
        setTestAccuracies([]);
        setLoggingMessages([]);
        setStatusMessage("");
        setErrorMessage("");
        messagesQueue = [];
    }

    // REACT COMPONENTS TO HANDLE CHATBOX

  const [typing, setTyping] = React.useState(false);
  let trainingStatus = "BEFORE FINETUNING";
  const [messages, setMessages] = React.useState([
    {
      message: "Hello! Ask me some questions!",
      sender: `TinyLlama [${trainingStatus}]`,
    }
  ])

  const handleSend = (message) => {
    const newMessage = {
      message: message,
      sender: "You",
      direction: "outgoing"
    }

    // update messages state
    setMessages([...messages, newMessage])

    setTyping(true)
    // process message to chatbot
  }

  // TRAINING FUNCS
  async function train() {
    console.log("Training started");
  }

  return (
    <>
      <div className="App" style = {{width: "1500px", height: "1000px"}}>
        <Container fluid>
        {/* <Container> */}
          {/* CHATBOX ===============================================================================================*/}
          <Row>
              <Col>
          <MainContainer responsive style = {{height: "800px"}}>
            <ChatContainer>
              <MessageList
                typingIndicator={typing ? <TypingIndicator content ={ `TinyLlama [${trainingStatus}] is typing...` } /> : null}
              >
                { messages.map((message, i) => {
                  return <Message key={i} model={message}>
                    <Message.Header sender={message.sender} />
                    </Message>
                })
                }
              </MessageList>
              <MessageInput placeholder="Type a message..." onSend={handleSend} attachButton={false} />
            </ChatContainer>
          </MainContainer>
         </Col> 
          {/* TRAINING DASHBOARD ====================================================================================*/}
          <Col>
            <div className="section">
                <h3>Fine-tune the model</h3>
                <Grid container spacing={{ xs: 1, md: 2 }}>
                    <Grid item xs={12} md={4} >
                        <TextField label="Number of epochs"
                            type="number"
                            value={numEpochs}
                            onChange={(e) => setNumEpochs(Number(e.target.value))}
                        />
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <TextField label="Batch size"
                            type="number"
                            value={batchSize}
                            onChange={(e) => setBatchSize(Number(e.target.value))}
                        />
                    </Grid>
                </Grid>
            </div>
            <br/>
            <div className="section">
                <Grid container spacing={{ xs: 1, md: 2 }}>
                    <Grid item xs={12} md={4} >
                        <TextField type="number"
                            label="Max number of training samples"
                            value={maxNumTrainSamples}
                            onChange={(e) => setMaxNumTrainSamples(Number(e.target.value))}
                        />
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <TextField type="number"
                            label="Max number of test samples"
                            value={maxNumTestSamples}
                            onChange={(e) => setMaxNumTestSamples(Number(e.target.value))}
                        />
                    </Grid>
                </Grid>

            </div>
            <div className="section">
                <FormControlLabel
                    control={<Switch
                        checked={enableLiveLogging}
                        onChange={(e) => setEnableLiveLogging(!enableLiveLogging)} />}
                    label='Log all batch results as they happen. Can slow down training.' />
            </div>
            <div className="section">
                <Button onClick={train}
                    disabled={isTraining}
                    variant='contained'>
                    Train
                </Button>
                <br></br>
            </div>
            <pre>{statusMessage}</pre>
            {errorMessage &&
                <p className='error'>
                    {errorMessage}
                </p>}

            {loggingMessages.length > 0 &&
                <div>
                    <h3>Logs:</h3>
                    <pre>
                        {loggingMessages.map((m, i) => (<React.Fragment key={i}>
                            {m}
                            <br />
                        </React.Fragment>))}
                    </pre>
                </div>}
          </Col>
         </Row>
          </Container>
      </div>
    </>
  )
}

export default App
