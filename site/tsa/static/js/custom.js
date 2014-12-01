function loadAction(){
    // disable or enable fields
    $("#analysis_form #id_saved_ais").prop('disabled', false);
    $("#analysis_form #id_ai_types").prop('disabled', true);
    $("#analysis_form #id_tweet_sets").prop('disabled', true);
    $("#analysis_form #analysis_form #id_name").prop('disabled', true);
    $("#analysis_form #id_custom_tweet_set").prop('disabled', true);
    $("#analysis_form #id_save_results").prop('disabled', true);
    // hide or show form groups
    $("#analysis_form #saved_ai_form").show();
    $("#analysis_form #ai_types_form").hide();
    $("#analysis_form #tweet_sets_form").hide();
    $("#analysis_form #custom_tweet_set_div").hide();
    $("#analysis_form #save_results_div").hide();
}

function createAction(){
    // disable or enable fields
    $("#analysis_form #id_saved_ais").prop('disabled', true);
    $("#analysis_form #id_ai_types").prop('disabled', false);
    $("#analysis_form #id_tweet_sets").prop('disabled', false);
    $("#analysis_form #analysis_form #id_name").prop('disabled', false);
    $("#analysis_form #id_custom_tweet_set").prop('disabled', false);
    $("#analysis_form #id_save_results").prop('disabled', false);
    // hide or show form groups
    $("#analysis_form #saved_ai_form").hide();
    $("#analysis_form #ai_types_form").show();
    $("#analysis_form #tweet_sets_form").show();
    $("#analysis_form #custom_tweet_set_div").show();
    $("#analysis_form #save_results_div").show();
}

function customTweetSetShow()
{
    $("#analysis_form label[for='id_tweet_sets']").show();
    $("#analysis_form #id_tweet_sets").show();
    $("#analysis_form #id_tweet_sets").prop('disabled', false);
}

function customTweetSetHide()
{
    $("#analysis_form label[for='id_tweet_sets']").hide();
    $("#analysis_form #id_tweet_sets").hide();
    $("#analysis_form #id_tweet_sets").prop('disabled', true);
}

function saveResultsShow()
{
    $("#analysis_form label[for='id_name']").show();
    $("#analysis_form #id_name").show();
    $("#analysis_form #id_name").prop('disabled', false);
}

function saveResultsHide()
{
    $("#analysis_form label[for='id_name']").hide();
    $("#analysis_form #id_name").hide();
    $("#analysis_form #id_name").prop('disabled', true);
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

function manageFields(){
  if($("#id_action_0").is(':checked')) {
        createAction();
  } else if($("#id_action_1").is(':checked')){
        loadAction();
  }
  if($("#id_custom_tweet_set").is(':checked')) {
        customTweetSetShow();
  } else {
        customTweetSetHide();
  }
  if($("#id_save_results").is(':checked')) {
        saveResultsShow();
  } else {
        saveResultsHide();
  }
}

$(document).ready(function () {
        manageFields();
        var url = window.location;
        $('ul.nav a[href="'+ url +'"]').parent().addClass('active');
        $('ul.nav a').filter(function() {
             return this.href == url;
        }).parent().addClass('active');
    });
