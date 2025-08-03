import React, { useState } from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  TextField,
  Box,
  Chip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import {
  Phone as PhoneIcon,
  Email as EmailIcon,
  LocationOn as LocationIcon,
  Send as SendIcon,
  ContactSupport as ContactIcon,
  Warning as EmergencyIcon,
} from '@mui/icons-material';
import { useMutation } from 'react-query';
import axios from 'axios';

const Authorities = () => {
  const [location, setLocation] = useState('');
  const [disasterType, setDisasterType] = useState('');
  const [authorities, setAuthorities] = useState([]);
  const [selectedAuthority, setSelectedAuthority] = useState(null);
  const [contactDialog, setContactDialog] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const searchMutation = useMutation(
    async (data) => {
      const response = await axios.get('/api/authorities', { params: data });
      return response.data;
    },
    {
      onSuccess: (data) => {
        setAuthorities(data.authorities || []);
        setError('');
      },
      onError: (error) => {
        setError(error.response?.data?.error || 'Error fetching authorities');
        setAuthorities([]);
      },
    }
  );

  const contactMutation = useMutation(
    async (data) => {
      const response = await axios.post('/api/contact-authority', data);
      return response.data;
    },
    {
      onSuccess: (data) => {
        setContactDialog(false);
        setMessage('');
        setSelectedAuthority(null);
        // Show success message
        alert('Alert sent successfully!');
      },
      onError: (error) => {
        alert(error.response?.data?.error || 'Error sending alert');
      },
    }
  );

  const handleSearch = () => {
    if (!location.trim()) {
      setError('Please enter a location');
      return;
    }

    searchMutation.mutate({
      location: location.trim(),
      disaster_type: disasterType || undefined,
    });
  };

  const handleContact = (authority) => {
    setSelectedAuthority(authority);
    setContactDialog(true);
  };

  const handleSendAlert = () => {
    if (!message.trim()) {
      alert('Please enter a message');
      return;
    }

    contactMutation.mutate({
      authority_id: selectedAuthority.id,
      message: message.trim(),
      location: location,
      disaster_type: disasterType,
    });
  };

  const disasterTypes = [
    { value: 'wildfire', label: 'Wildfire', color: 'error' },
    { value: 'flood', label: 'Flood', color: 'info' },
    { value: 'hurricane', label: 'Hurricane', color: 'warning' },
    { value: 'earthquake', label: 'Earthquake', color: 'secondary' },
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Emergency Authorities
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Find and contact relevant emergency services based on location and disaster type
      </Typography>

      {/* Search Section */}
      <Card elevation={3} sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Search Authorities
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Location"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="e.g., New York, NY"
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                select
                label="Disaster Type (Optional)"
                value={disasterType}
                onChange={(e) => setDisasterType(e.target.value)}
                variant="outlined"
              >
                <option value="">All Types</option>
                {disasterTypes.map((type) => (
                  <option key={type.value} value={type.value}>
                    {type.label}
                  </option>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12}>
              <Button
                variant="contained"
                startIcon={<ContactIcon />}
                onClick={handleSearch}
                disabled={searchMutation.isLoading}
                fullWidth
              >
                {searchMutation.isLoading ? 'Searching...' : 'Search Authorities'}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Results Section */}
      {authorities.length > 0 && (
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Found {authorities.length} Authority{authorities.length !== 1 ? 'ies' : ''}
          </Typography>
          
          <Grid container spacing={3}>
            {authorities.map((authority, index) => (
              <Grid item xs={12} md={6} key={index}>
                <Card elevation={2}>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                      <Typography variant="h6" component="div">
                        {authority.name}
                      </Typography>
                      <Chip
                        label={authority.type}
                        color="primary"
                        size="small"
                      />
                    </Box>
                    
                    <List dense>
                      {authority.phone && (
                        <ListItem sx={{ py: 0 }}>
                          <ListItemIcon>
                            <PhoneIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText
                            primary="Phone"
                            secondary={authority.phone}
                          />
                        </ListItem>
                      )}
                      
                      {authority.email && (
                        <ListItem sx={{ py: 0 }}>
                          <ListItemIcon>
                            <EmailIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText
                            primary="Email"
                            secondary={authority.email}
                          />
                        </ListItem>
                      )}
                      
                      {authority.distance && (
                        <ListItem sx={{ py: 0 }}>
                          <ListItemIcon>
                            <LocationIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText
                            primary="Distance"
                            secondary={authority.distance}
                          />
                        </ListItem>
                      )}
                      
                      {authority.response_time && (
                        <ListItem sx={{ py: 0 }}>
                          <ListItemIcon>
                            <EmergencyIcon color="primary" />
                          </ListItemIcon>
                          <ListItemText
                            primary="Response Time"
                            secondary={authority.response_time}
                          />
                        </ListItem>
                      )}
                    </List>
                    
                    <Box sx={{ mt: 2 }}>
                      <Button
                        variant="contained"
                        startIcon={<SendIcon />}
                        onClick={() => handleContact(authority)}
                        fullWidth
                      >
                        Send Alert
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* Contact Dialog */}
      <Dialog open={contactDialog} onClose={() => setContactDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          Send Alert to {selectedAuthority?.name}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Authority: {selectedAuthority?.name}
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Type: {selectedAuthority?.type}
            </Typography>
            {location && (
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Location: {location}
              </Typography>
            )}
            {disasterType && (
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Disaster Type: {disasterType}
              </Typography>
            )}
          </Box>
          
          <TextField
            fullWidth
            multiline
            rows={4}
            label="Alert Message"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Enter your emergency alert message..."
            variant="outlined"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setContactDialog(false)}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleSendAlert}
            disabled={contactMutation.isLoading}
          >
            {contactMutation.isLoading ? 'Sending...' : 'Send Alert'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Information Section */}
      <Card elevation={2}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            About Emergency Authorities
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            This system helps you find and contact relevant emergency services based on your location and the type of disaster. 
            When you send an alert, it will be delivered to the appropriate authorities through multiple channels including email, 
            SMS, and phone calls for critical situations.
          </Typography>
          
          <Typography variant="subtitle2" gutterBottom>
            Available Authority Types:
          </Typography>
          <Grid container spacing={1} sx={{ mt: 1 }}>
            {disasterTypes.map((type) => (
              <Grid item key={type.value}>
                <Chip
                  label={type.label}
                  color={type.color}
                  size="small"
                  variant="outlined"
                />
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
    </Container>
  );
};

export default Authorities; 