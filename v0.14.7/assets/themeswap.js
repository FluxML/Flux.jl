// Small function to quickly swap out themes. Gets put into the <head> tag..
function set_theme_from_local_storage() {
  // Intialize the theme to null, which means default
  var theme = null;
  // If the browser supports the localstorage and is not disabled then try to get the
  // documenter theme
  if(window.localStorage != null) {
    // Get the user-picked theme from localStorage. May be `null`, which means the default
    // theme.
    theme =  window.localStorage.getItem("documenter-theme");
  }
  // Check if the browser supports user color preference
  var darkPreference = false;
  // Check if the users preference is for dark color scheme
  if(window.matchMedia('(prefers-color-scheme: dark)').matches === true) {
    darkPreference = true;
  }
  // Initialize a few variables for the loop:
  //
  //  - active: will contain the index of the theme that should be active. Note that there
  //    is no guarantee that localStorage contains sane values. If `active` stays `null`
  //    we either could not find the theme or it is the default (primary) theme anyway.
  //    Either way, we then need to stick to the primary theme.
  //
  //  - disabled: style sheets that should be disabled (i.e. all the theme style sheets
  //    that are not the currently active theme)
  var active = null; var disabled = []; var darkTheme = null;
  for (var i = 0; i < document.styleSheets.length; i++) {
    var ss = document.styleSheets[i];
    // The <link> tag of each style sheet is expected to have a data-theme-name attribute
    // which must contain the name of the theme. The names in localStorage much match this.
    var themename = ss.ownerNode.getAttribute("data-theme-name");
    // attribute not set => non-theme stylesheet => ignore
    if(themename === null) continue;
    // To distinguish the default (primary) theme, it needs to have the data-theme-primary
    // attribute set.
    var isprimary = (ss.ownerNode.getAttribute("data-theme-primary") !== null);
    // Check if the theme is primary dark theme
    var isDarkTheme = (ss.ownerNode.getAttribute("data-theme-primary-dark") !== null);
    // If ss is for dark theme then set the value of darkTheme to the name of the theme
    if(isDarkTheme) darkTheme = themename;
    // If we find a matching theme (and it's not the default), we'll set active to non-null
    if(themename === theme) active = i;
    // Store the style sheets of inactive themes so that we could disable them
    if(themename !== theme) disabled.push(ss);
  }
  if(active !== null) {
    // If we did find an active theme, we'll (1) add the theme--$(theme) class to <html>
    document.getElementsByTagName('html')[0].className = "theme--" + theme;
    // and (2) disable all the other theme stylesheets
    disabled.forEach(function(ss){
      ss.disabled = true;
    });
  }
  else if(darkTheme !== null && darkPreference === true) {
    // If we did find an active theme, we'll (1) add the theme--$(theme) class to <html>
    document.getElementsByTagName('html')[0].className = "theme--" + darkTheme;
    // and (2) disable all the other theme stylesheets
    disabled.forEach(function(ss){
      if (ss.ownerNode.getAttribute("data-theme-name") !== darkTheme) {
        ss.disabled = true;
      }
    });
  }
}
set_theme_from_local_storage();
