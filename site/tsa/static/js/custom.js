function loadAction(){
    // disable or enable fields
    $("#id_saved_ais").prop('disabled', false);
    $("#id_ai_types").prop('disabled', true);
    $("#id_tweet_sets").prop('disabled', true);
    $("#id_name").prop('disabled', true);
    $("#id_custom_tweet_set").prop('disabled', true);
    $("#id_save_results").prop('disabled', true);
    // hide or show form groups
    $("#saved_ai_form").show();
    $("#ai_types_form").hide();
    $("#tweet_sets_form").hide();
    $("#custom_tweet_set_div").hide();
    $("#save_results_div").hide();
}

function createAction(){
    // disable or enable fields
    $("#id_saved_ais").prop('disabled', true);
    $("#id_ai_types").prop('disabled', false);
    $("#id_tweet_sets").prop('disabled', false);
    $("#id_name").prop('disabled', false);
    $("#id_custom_tweet_set").prop('disabled', false);
    $("#id_save_results").prop('disabled', false);
    // hide or show form groups
    $("#saved_ai_form").hide();
    $("#ai_types_form").show();
    $("#tweet_sets_form").show();
    $("#custom_tweet_set_div").show();
    $("#save_results_div").show();
}

function customTweetSetShow()
{
    $("label[for='id_tweet_sets']").show();
    $("#id_tweet_sets").show();
    $("#id_tweet_sets").prop('disabled', false);
}

function customTweetSetHide()
{
    $("label[for='id_tweet_sets']").hide();
    $("#id_tweet_sets").hide();
    $("#id_tweet_sets").prop('disabled', true);
}

function saveResultsShow()
{
    $("label[for='id_name']").show();
    $("#id_name").show();
    $("#id_name").prop('disabled', false);
}

function saveResultsHide()
{
    $("label[for='id_name']").hide();
    $("#id_name").hide();
    $("#id_name").prop('disabled', true);
}

$("#id_action_0").change(function(){
    if($(this).is(':checked')) {
        createAction();
    }
});

$("#id_action_1").change(function(){
    if($(this).is(':checked')) {
        loadAction();
    }
});

$("#id_custom_tweet_set").change(function(){
    if($(this).is(':checked')) {
        customTweetSetShow();
    } else {
        customTweetSetHide();
    }
});

$("#id_save_results").change(function(){
    if($(this).is(':checked')) {
        saveResultsShow();
    } else {
        saveResultsHide();
    }
});

$(document).ready(function () {
        hideBackArrowIfInPageOne();
        createAction();
        var url = window.location;
        $('ul.nav a[href="'+ url +'"]').parent().addClass('active');
        $('ul.nav a').filter(function() {
             return this.href == url;
        }).parent().addClass('active');
    });
