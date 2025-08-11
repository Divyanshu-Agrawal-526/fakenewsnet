import React, { useMemo, useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  Box,
  Card,
  CardContent,
  Divider,
  Chip,
  Alert,
} from '@mui/material';
import { useMutation } from 'react-query';
import axios from 'axios';

const prettyJson = (obj) => JSON.stringify(obj, null, 2);

const parseTextsFromTextarea = (value) =>
  value
    .split('\n')
    .map((t) => t.trim())
    .filter((t) => t.length > 0);

function Integrated() {
  const [rawTexts, setRawTexts] = useState('');
  const [modelPath, setModelPath] = useState('');
  const [processResult, setProcessResult] = useState(null);
  const [summaryResult, setSummaryResult] = useState(null);
  const [subscribeTopicId, setSubscribeTopicId] = useState('');
  const [subscribeLocation, setSubscribeLocation] = useState('');
  const [subscribeResult, setSubscribeResult] = useState(null);
  const [eventDir, setEventDir] = useState('');
  const [split, setSplit] = useState('train');
  const [errorMessage, setErrorMessage] = useState('');

  const textsArray = useMemo(() => parseTextsFromTextarea(rawTexts), [rawTexts]);

  const processMutation = useMutation(
    async () => {
      const payload = { texts: textsArray };
      if (modelPath) payload.load_detector_path = modelPath;
      const { data } = await axios.post('/api/integrated/process', payload);
      return data;
    },
    {
      onSuccess: (data) => {
        setProcessResult(data);
        setErrorMessage('');
      },
      onError: (err) => {
        setErrorMessage(err.response?.data?.error || 'Failed to process texts');
        setProcessResult(null);
      },
    }
  );

  const summaryMutation = useMutation(
    async () => {
      const { data } = await axios.get('/api/integrated/summary');
      return data;
    },
    {
      onSuccess: (data) => {
        setSummaryResult(data);
        setErrorMessage('');
      },
      onError: (err) => {
        setErrorMessage(err.response?.data?.error || 'Failed to fetch summary');
        setSummaryResult(null);
      },
    }
  );

  const subscribeMutation = useMutation(
    async () => {
      const params = new URLSearchParams();
      params.set('topic_id', String(parseInt(subscribeTopicId || '0', 10)));
      if (subscribeLocation) params.set('location', subscribeLocation);
      const { data } = await axios.get(`/api/integrated/subscribe?${params.toString()}`);
      return data;
    },
    {
      onSuccess: (data) => {
        setSubscribeResult(data);
        setErrorMessage('');
      },
      onError: (err) => {
        setErrorMessage(err.response?.data?.error || 'Failed to subscribe');
        setSubscribeResult(null);
      },
    }
  );

  const crisisMutation = useMutation(
    async () => {
      const payload = { event_dir: eventDir, split: split || 'train' };
      const { data } = await axios.post('/api/integrated/process-crisisnlp', payload);
      return data;
    },
    {
      onSuccess: (data) => {
        setProcessResult(data);
        setErrorMessage('');
      },
      onError: (err) => {
        setErrorMessage(err.response?.data?.error || 'Failed to process CrisisNLP event');
      },
    }
  );

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>
        Integrated Research Model
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Run topic modeling, community detection, geolocation extraction, and optional misinformation filtering,
        as described in the research system. Provide multiple texts (one per line), optionally load a detector,
        and view summaries or subscribe to topics.
      </Typography>

      {errorMessage && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {errorMessage}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Input Texts
              </Typography>
              <TextField
                label="Enter texts (one per line)"
                multiline
                minRows={6}
                fullWidth
                value={rawTexts}
                onChange={(e) => setRawTexts(e.target.value)}
              />
              <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
                <TextField
                  label="Optional model path (load_detector_path)"
                  fullWidth
                  placeholder="models/saved_models/fn_bert_tfidf"
                  value={modelPath}
                  onChange={(e) => setModelPath(e.target.value)}
                />
              </Box>

              <Box sx={{ display: 'flex', gap: 2, mt: 2, flexWrap: 'wrap' }}>
                <Button
                  variant="contained"
                  onClick={() => processMutation.mutate()}
                  disabled={textsArray.length === 0 || processMutation.isLoading}
                >
                  {processMutation.isLoading ? 'Processing…' : 'Run Integrated Process'}
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => summaryMutation.mutate()}
                  disabled={summaryMutation.isLoading}
                >
                  {summaryMutation.isLoading ? 'Fetching…' : 'Get Summary'}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Subscribe to Topic
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  label="Topic ID"
                  type="number"
                  value={subscribeTopicId}
                  onChange={(e) => setSubscribeTopicId(e.target.value)}
                />
                <TextField
                  label="Location filter (optional)"
                  value={subscribeLocation}
                  onChange={(e) => setSubscribeLocation(e.target.value)}
                />
                <Button
                  variant="outlined"
                  onClick={() => subscribeMutation.mutate()}
                  disabled={subscribeMutation.isLoading || subscribeTopicId === ''}
                >
                  {subscribeMutation.isLoading ? 'Subscribing…' : 'Subscribe'}
                </Button>
              </Box>
            </CardContent>
          </Card>

          <Box sx={{ height: 16 }} />

          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Process CrisisNLP Event
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  label="Event directory"
                  placeholder="data/crisisnlp_dataset/events_set2/kerala_floods_2018"
                  value={eventDir}
                  onChange={(e) => setEventDir(e.target.value)}
                />
                <TextField
                  label="Split (train|dev|test)"
                  value={split}
                  onChange={(e) => setSplit(e.target.value)}
                />
                <Button
                  variant="outlined"
                  onClick={() => crisisMutation.mutate()}
                  disabled={!eventDir || crisisMutation.isLoading}
                >
                  {crisisMutation.isLoading ? 'Processing…' : 'Run CrisisNLP'}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          {(processResult || summaryResult || subscribeResult) && (
            <Paper elevation={2} sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Results
              </Typography>

              {processResult && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Processing
                  </Typography>
                  <Divider sx={{ mb: 1 }} />
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 1 }}>
                    {Array.isArray(processResult?.processing_steps) &&
                      processResult.processing_steps.map((step) => (
                        <Chip key={step} label={step} color="primary" variant="outlined" />
                      ))}
                  </Box>
                  <pre style={{ margin: 0 }}>{prettyJson(processResult)}</pre>
                </Box>
              )}

              {summaryResult && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Summary
                  </Typography>
                  <Divider sx={{ mb: 1 }} />
                  <pre style={{ margin: 0 }}>{prettyJson(summaryResult)}</pre>
                </Box>
              )}

              {subscribeResult && (
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    Subscription Updates
                  </Typography>
                  <Divider sx={{ mb: 1 }} />
                  <pre style={{ margin: 0 }}>{prettyJson(subscribeResult)}</pre>
                </Box>
              )}
            </Paper>
          )}
        </Grid>
      </Grid>
    </Container>
  );
}

export default Integrated;





