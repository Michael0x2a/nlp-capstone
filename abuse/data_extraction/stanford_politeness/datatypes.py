from arrow import Arrow

Community = str
RequestId = str
Username = str
AnnotatorId = str
Timestamp = Arrow

class Request:
    def __init__(self, community: Community,
                       request_id: RequestId,
                       text: str,
                       timestamp: Timestamp,
                       user: Username) -> None:
        self.community = community
        self.request_id = request_id
        self.text = text
        self.timestamp = timestamp
        self.user = user


class AnnotatedRequest:
    def __init__(self, community: Community,
                       request_id: RequestId,
                       timestamp: Timestamp,
                       scores: List[int],
                       annotator_ids: List[AnnotatorId],
                       normalized_score: float) -> None:
        self.community = community
        self.request_id = request_id
        self.text = text
        self.timestamp = timestamp
        self.user = user

class StackOverflowRequest(Request):
    def __init__(self, community: Community,
                       request_id: RequestId,
                       text: str,
                       timestamp: Timestamp,
                       user: Username,
                       reputation: int,
                       upvotes: int,
                       downvotes: int) -> None:
        super().__init__(community, request_id, text, timestamp, user)

        self.reputation = reputation
        self.upvotes = upvotes
        self.downvotes = downvotes

class StackOverflowRole:
    def __init__(self, community: Community,
                       request_id: RequestId,
                       user_id: Username,
                       post_id: str,
                       author_role: str) -> None:
        self.community = community
        self.request_id = request_id
        self.post_id = post_id
        self.author_role = author_role


