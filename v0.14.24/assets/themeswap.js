// Small function to quickly swap out themes. Gets put into the <head> tag..
function set_theme_from_local_storage() {
  // Initialize the theme to null, which means default
  var theme = null;
  // If the browser supports the localstorage and is not disabled then try to get the
  // documenter theme
  if (window.localStorage != null) {
    // Get the user-picked theme from localStorage. May be `null`, which means the default
    // theme.
    theme = window.localStorage.getItem("documenter-theme");
  }
  // Check if the users preference is for dark color scheme
  var darkPreference =
    window.matchMedia("(prefers-color-scheme: dark)").matches === true;
  // Initialize a few variables for the loop:
  //
  //  - active: will contain the index of the theme that should be active. Note that there
  //    is no guarantee that localStorage contains sane values. If `active` stays `null`
  //    we either could not find the theme or it is the default (primary) theme anyway.
  //    Either way, we then need to stick to the primary theme.
  //
  //  - disabled: style sheets that should be disabled (i.e. all the theme style sheets
  //    that are not the currently active theme)
  var active = null;
  var disabled = [];
  var primaryLightTheme = null;
  var primaryDarkTheme = null;
  for (var i = 0; i < document.styleSheets.length; i++) {
    var ss = document.styleSheets[i];
    // The <link> tag of each style sheet is expected to have a data-theme-name attribute
    // which must contain the name of the theme. The names in localStorage much match this.
    var themename = ss.ownerNode.getAttribute("data-theme-name");
    // attribute not set => non-theme stylesheet => ignore
    if (themename === null) continue;
    // To distinguish the default (primary) theme, it needs to have the data-theme-primary
    // attribute set.
    if (ss.ownerNode.getAttribute("data-theme-primary") !== null) {
      primaryLightTheme = themename;
    }
    // Check if the theme is primary dark theme so that we could store its name in darkTheme
    if (ss.ownerNode.getAttribute("data-theme-primary-dark") !== null) {
      primaryDarkTheme = themename;
    }
    // If we find a matching theme (and it's not the default), we'll set active to non-null
    if (themename === theme) active = i;
    // Store the style sheets of inactive themes so that we could disable them
    if (themename !== theme) disabled.push(ss);
  }
  var activeTheme = null;
  if (active !== null) {
    // If we did find an active theme, we'll (1) add the theme--$(theme) class to <html>
    document.getElementsByTagName("html")[0].className = "theme--" + theme;
    activeTheme = theme;
  } else {
    // If we did _not_ find an active theme, then we need to fall back to the primary theme
    // which can either be dark or light, depending on the user's OS preference.
    var activeTheme = darkPreference ? primaryDarkTheme : primaryLightTheme;
    // In case it somehow happens that the relevant primary theme was not found in the
    // preceding loop, we abort without doing anything.
    if (activeTheme === null) {
      console.error("Unable to determine primary theme.");
      return;
    }
    // When switching to the primary light theme, then we must not have a class name
    // for the <html> tag. That's only for non-primary or the primary dark theme.
    if (darkPreference) {
      document.getElementsByTagName("html")[0].className =
        "theme--" + activeTheme;
    } else {
      document.getElementsByTagName("html")[0].className = "";
    }
  }
  for (var i = 0; i < document.styleSheets.length; i++) {
    var ss = document.styleSheets[i];
    // The <link> tag of each style sheet is expected to have a data-theme-name attribute
    // which must contain the name of the theme. The names in localStorage much match this.
    var themename = ss.ownerNode.getAttribute("data-theme-name");
    // attribute not set => non-theme stylesheet => ignore
    if (themename === null) continue;
    // we'll disable all the stylesheets, except for the active one
    ss.disabled = !(themename == activeTheme);
  }
}
set_theme_from_local_storage();
