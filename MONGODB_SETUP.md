# MongoDB Setup for ChatBOT.EDU

## Overview
ChatBOT.EDU now supports MongoDB for persistent conversation storage, providing better scalability and data management compared to JSON file storage.

## Features
- ✅ **Conversation Persistence**: All conversations are saved to MongoDB
- ✅ **Message History**: Complete message history with timestamps
- ✅ **Image Storage**: Images are stored in MongoDB with proper metadata
- ✅ **User Tracking**: Anonymous user sessions for conversation isolation
- ✅ **Fallback Support**: Falls back to JSON file storage if MongoDB is unavailable

## Setup Instructions

### 1. MongoDB Atlas Setup (Recommended)
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a free cluster
3. Create a database user with read/write permissions
4. Get your connection string

### 2. Environment Variables
Set the following environment variables:

```bash
# Required
MONGODB_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
MONGODB_DB_NAME=chatbot_edu

# Optional (if not set, defaults to "chatbot_edu")
MONGODB_DB_NAME=your_database_name
```

### 3. Streamlit Secrets (Alternative)
Add to `.streamlit/secrets.toml`:

```toml
MONGODB_URI = "mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGODB_DB_NAME = "chatbot_edu"
```

### 4. Install Dependencies
```bash
pip install pymongo==4.10.1
```

## Database Schema

### Collections Created:
- **conversations**: Conversation metadata and settings
- **messages**: Individual messages with content and timestamps
- **message_images**: Image data and metadata
- **user_sessions**: Anonymous user session tracking

### Example Document Structure:

#### conversations
```json
{
  "_id": ObjectId("..."),
  "created_at": "2024-01-01T12:00:00.000Z",
  "updated_at": "2024-01-01T12:30:00.000Z",
  "model": "gpt-5-mini",
  "name": "AI Chat Discussion",
  "user_hash": "abc123def456",
  "session_id": "session_xyz789",
  "message_count": 10,
  "enable_internet": false,
  "enable_documents": true,
  "enable_images": true
}
```

#### messages
```json
{
  "_id": ObjectId("..."),
  "conversation_id": ObjectId("..."),
  "role": "user",
  "content": "Hello, how are you?",
  "ts": "2024-01-01T12:00:00.000Z",
  "user_hash": "abc123def456",
  "has_images": false
}
```

## Usage

### Automatic Operation
Once configured, MongoDB integration works automatically:
- New conversations are saved to MongoDB
- All messages are stored with full metadata
- Images are stored with proper MIME types
- Conversation summaries are updated in real-time

### Fallback Behavior
If MongoDB is unavailable:
- Application continues to work normally
- Falls back to JSON file storage
- No data loss occurs

### Data Analysis
You can query MongoDB directly for analytics:

```javascript
// Count total conversations
db.conversations.countDocuments({})

// Get conversations by date range
db.conversations.find({
  "created_at": {
    $gte: "2024-01-01T00:00:00.000Z",
    $lt: "2024-02-01T00:00:00.000Z"
  }
})

// Get message statistics
db.messages.aggregate([
  { $group: { _id: "$role", count: { $sum: 1 } } }
])

// Get conversations with images
db.conversations.find({ "image_count": { $gt: 0 } })
```

## Security Notes
- Use MongoDB Atlas IP whitelist for production
- Enable authentication and authorization
- Consider using MongoDB's built-in encryption
- Regular backups recommended for production use

## Troubleshooting

### Connection Issues
- Check MongoDB URI format
- Verify network connectivity
- Ensure database user has proper permissions

### Performance
- MongoDB Atlas free tier supports up to 512MB
- Consider upgrading for high-volume usage
- Indexes are automatically created for optimal performance

## Migration from JSON
If you have existing conversations in JSON format:
1. The application will automatically load from JSON if MongoDB is unavailable
2. New conversations will be saved to MongoDB
3. Existing conversations remain in JSON until manually migrated
