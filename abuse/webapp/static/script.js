$(document).ready(function() {
    var CLASSIFY_URL = "/api/classify";
    
    function load(div_id, blob) {
        var prob = blob['prob_attack'].toFixed(6);
        $(div_id).html(prob);
        $(div_id).removeClass("prob-ok prob-neutral prob-bad");
        if (prob < 0.4) {
            $(div_id).addClass("prob-ok");
        } else if (0.4 <= prob && prob <= 0.6) {
            $(div_id).addClass("prob-neutral");
        } else {
            $(div_id).addClass("prob-bad");
        }
    }

    function handleClassified(data) {
        console.log("Got data!");
        console.log(data);
    
        load('#profanity-attack', data['attack']['profanity'])
        load('#bag-attack', data['attack']['bag_of_words']);
        load('#lr-attack', data['attack']['lr']);
        load('#rnn-attack', data['attack']['rnn']);

        load('#profanity-aggression', data['aggression']['profanity'])
        load('#bag-aggression', data['aggression']['bag_of_words']);
        load('#lr-aggression', data['aggression']['lr']);
        load('#rnn-aggression', data['aggression']['rnn']);

        load('#profanity-toxicity', data['toxicity']['profanity'])
        load('#bag-toxicity', data['toxicity']['bag_of_words']);
        load('#lr-toxicity', data['toxicity']['lr']);
        load('#rnn-toxicity', data['toxicity']['rnn']);
    }

    // Document setup
    $('tbody div').html('N/A');

    $("#submit").click(function() {
        var comment = $("#comment-area").val();
        console.log(comment);
        $.ajax({
            "url": CLASSIFY_URL, 
            "type": "POST",
            "contentType": "application/json",
            "data": JSON.stringify({'comment': comment}),
            "dataType": "json"
        }).done(handleClassified);
    });
});
