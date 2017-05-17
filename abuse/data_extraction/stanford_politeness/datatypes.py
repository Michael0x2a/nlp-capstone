from typing import List, TypeVar, Any
import arrow

Community = str
RequestId = str
Username = str
AnnotatorId = str
Timestamp = arrow.Arrow

T = TypeVar('T', bound='Annotation')

class Annotation:
    @classmethod
    def from_row(cls, args: List[str]) -> Any:
        raise NotImplementedError();

class Request(Annotation):
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

    @classmethod
    def from_row(cls, args: List[str]) -> 'Request':
        return Request(args[0], args[1], args[2], arrow.get(args[3]), args[4])

class AnnotatedRequest(Annotation):
    def __init__(self, community: Community,
                       request_id: RequestId,
                       text: str,
                       scores: List[int],
                       annotator_ids: List[AnnotatorId],
                       normalized_score: float) -> None:
        self.community = community
        self.request_id = request_id
        self.text = text
        self.scores = scores
        self.annotator_ids = annotator_ids
        self.normalized_score = normalized_score

    @classmethod
    def from_row(cls, args: List[str]) -> 'AnnotatedRequest':
        return AnnotatedRequest(
                args[0],
                args[1],
                args[2],
                [int(x) for x in args[3:8]],
                args[8:13],
                float(args[13]))

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

    @classmethod
    def from_row(cls, args: List[str]) -> 'Request':
        return StackOverflowRequest(args[0], args[1], args[2], arrow.get(args[3]),
                args[4], int(args[5]), int(args[6]), int(args[7]))



