<!DOCTYPE html>
<!-- saved from url=(0049)http://localhost:8888/notebooks/data%20type.ipynb -->
<html lang="en-us"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    

    <title>data type - Jupyter Notebook</title>
    
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <link rel="stylesheet" href="./elandhiya_files/jquery-ui.min.css" type="text/css">
    <link rel="stylesheet" href="./elandhiya_files/jquery.typeahead.min.css" type="text/css">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    


<script type="text/javascript" src="./elandhiya_files/MathJax.js.download" charset="utf-8"></script>

<script type="text/javascript">
// MathJax disabled, set as null to distinguish from *missing* MathJax,
// where it will be undefined, and should prompt a dialog later.
window.mathjax_url = "/static/components/MathJax/MathJax.js";
</script>

<link rel="stylesheet" href="./elandhiya_files/bootstrap-tour.min.css" type="text/css">
<link rel="stylesheet" href="./elandhiya_files/codemirror.css">


    <link rel="stylesheet" href="./elandhiya_files/style.min.css" type="text/css">
    

<link rel="stylesheet" href="./elandhiya_files/override.css" type="text/css">
<link rel="stylesheet" href="http://localhost:8888/notebooks/data%20type.ipynb" id="kernel-css" type="text/css">


    <link rel="stylesheet" href="./elandhiya_files/custom.css" type="text/css">
    <script src="./elandhiya_files/promise.min.js.download" type="text/javascript" charset="utf-8"></script>
    <script src="./elandhiya_files/react.production.min.js.download" type="text/javascript"></script>
    <script src="./elandhiya_files/react-dom.production.min.js.download" type="text/javascript"></script>
    <script src="./elandhiya_files/index.js.download" type="text/javascript"></script>
    <script src="./elandhiya_files/require.js.download" type="text/javascript" charset="utf-8"></script>
    <script>
      require.config({
          
          urlArgs: "v=20230926153353",
          
          baseUrl: '/static/',
          paths: {
            'auth/js/main': 'auth/js/main.min',
            custom : '/custom',
            nbextensions : '/nbextensions',
            kernelspecs : '/kernelspecs',
            underscore : 'components/underscore/underscore-min',
            backbone : 'components/backbone/backbone-min',
            jed: 'components/jed/jed',
            jquery: 'components/jquery/jquery.min',
            json: 'components/requirejs-plugins/src/json',
            text: 'components/requirejs-text/text',
            bootstrap: 'components/bootstrap/dist/js/bootstrap.min',
            bootstraptour: 'components/bootstrap-tour/build/js/bootstrap-tour.min',
            'jquery-ui': 'components/jquery-ui/dist/jquery-ui.min',
            moment: 'components/moment/min/moment-with-locales',
            codemirror: 'components/codemirror',
            termjs: 'components/xterm.js/xterm',
            typeahead: 'components/jquery-typeahead/dist/jquery.typeahead.min',
          },
          map: { // for backward compatibility
              "*": {
                  "jqueryui": "jquery-ui",
              }
          },
          shim: {
            typeahead: {
              deps: ["jquery"],
              exports: "typeahead"
            },
            underscore: {
              exports: '_'
            },
            backbone: {
              deps: ["underscore", "jquery"],
              exports: "Backbone"
            },
            bootstrap: {
              deps: ["jquery"],
              exports: "bootstrap"
            },
            bootstraptour: {
              deps: ["bootstrap"],
              exports: "Tour"
            },
            "jquery-ui": {
              deps: ["jquery"],
              exports: "$"
            }
          },
          waitSeconds: 30,
      });

      require.config({
          map: {
              '*':{
                'contents': 'services/contents',
              }
          }
      });

      // error-catching custom.js shim.
      define("custom", function (require, exports, module) {
          try {
              var custom = require('custom/custom');
              console.debug('loaded custom.js');
              return custom;
          } catch (e) {
              console.error("error loading custom.js", e);
              return {};
          }
      })

      // error-catching custom-preload.js shim.
      define("custom-preload", function (require, exports, module) {
          try {
              var custom = require('custom/custom-preload');
              console.debug('loaded custom-preload.js');
              return custom;
          } catch (e) {
              console.error("error loading custom-preload.js", e);
              return {};
          }
      })

    document.nbjs_translations = {"domain": "nbjs", "locale_data": {"nbjs": {"": {"domain": "nbjs"}}}};
    document.documentElement.lang = navigator.language.toLowerCase();
    </script>

    
    

<script type="text/javascript" charset="utf-8" async="" data-requirecontext="_" data-requiremodule="services/contents" src="./elandhiya_files/contents.js.download"></script><script type="text/javascript" charset="utf-8" async="" data-requirecontext="_" data-requiremodule="custom/custom-preload" src="./elandhiya_files/custom-preload.js.download"></script><style type="text/css">.MathJax_Hover_Frame {border-radius: .25em; -webkit-border-radius: .25em; -moz-border-radius: .25em; -khtml-border-radius: .25em; box-shadow: 0px 0px 15px #83A; -webkit-box-shadow: 0px 0px 15px #83A; -moz-box-shadow: 0px 0px 15px #83A; -khtml-box-shadow: 0px 0px 15px #83A; border: 1px solid #A6D ! important; display: inline-block; position: absolute}
.MathJax_Menu_Button .MathJax_Hover_Arrow {position: absolute; cursor: pointer; display: inline-block; border: 2px solid #AAA; border-radius: 4px; -webkit-border-radius: 4px; -moz-border-radius: 4px; -khtml-border-radius: 4px; font-family: 'Courier New',Courier; font-size: 9px; color: #F0F0F0}
.MathJax_Menu_Button .MathJax_Hover_Arrow span {display: block; background-color: #AAA; border: 1px solid; border-radius: 3px; line-height: 0; padding: 4px}
.MathJax_Hover_Arrow:hover {color: white!important; border: 2px solid #CCC!important}
.MathJax_Hover_Arrow:hover span {background-color: #CCC!important}
</style><style type="text/css">#MathJax_About {position: fixed; left: 50%; width: auto; text-align: center; border: 3px outset; padding: 1em 2em; background-color: #DDDDDD; color: black; cursor: default; font-family: message-box; font-size: 120%; font-style: normal; text-indent: 0; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; z-index: 201; border-radius: 15px; -webkit-border-radius: 15px; -moz-border-radius: 15px; -khtml-border-radius: 15px; box-shadow: 0px 10px 20px #808080; -webkit-box-shadow: 0px 10px 20px #808080; -moz-box-shadow: 0px 10px 20px #808080; -khtml-box-shadow: 0px 10px 20px #808080; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true')}
#MathJax_About.MathJax_MousePost {outline: none}
.MathJax_Menu {position: absolute; background-color: white; color: black; width: auto; padding: 2px; border: 1px solid #CCCCCC; margin: 0; cursor: default; font: menu; text-align: left; text-indent: 0; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; z-index: 201; box-shadow: 0px 10px 20px #808080; -webkit-box-shadow: 0px 10px 20px #808080; -moz-box-shadow: 0px 10px 20px #808080; -khtml-box-shadow: 0px 10px 20px #808080; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true')}
.MathJax_MenuItem {padding: 2px 2em; background: transparent}
.MathJax_MenuArrow {position: absolute; right: .5em; padding-top: .25em; color: #666666; font-size: .75em}
.MathJax_MenuActive .MathJax_MenuArrow {color: white}
.MathJax_MenuArrow.RTL {left: .5em; right: auto}
.MathJax_MenuCheck {position: absolute; left: .7em}
.MathJax_MenuCheck.RTL {right: .7em; left: auto}
.MathJax_MenuRadioCheck {position: absolute; left: 1em}
.MathJax_MenuRadioCheck.RTL {right: 1em; left: auto}
.MathJax_MenuLabel {padding: 2px 2em 4px 1.33em; font-style: italic}
.MathJax_MenuRule {border-top: 1px solid #CCCCCC; margin: 4px 1px 0px}
.MathJax_MenuDisabled {color: GrayText}
.MathJax_MenuActive {background-color: Highlight; color: HighlightText}
.MathJax_MenuDisabled:focus, .MathJax_MenuLabel:focus {background-color: #E8E8E8}
.MathJax_ContextMenu:focus {outline: none}
.MathJax_ContextMenu .MathJax_MenuItem:focus {outline: none}
#MathJax_AboutClose {top: .2em; right: .2em}
.MathJax_Menu .MathJax_MenuClose {top: -10px; left: -10px}
.MathJax_MenuClose {position: absolute; cursor: pointer; display: inline-block; border: 2px solid #AAA; border-radius: 18px; -webkit-border-radius: 18px; -moz-border-radius: 18px; -khtml-border-radius: 18px; font-family: 'Courier New',Courier; font-size: 24px; color: #F0F0F0}
.MathJax_MenuClose span {display: block; background-color: #AAA; border: 1.5px solid; border-radius: 18px; -webkit-border-radius: 18px; -moz-border-radius: 18px; -khtml-border-radius: 18px; line-height: 0; padding: 8px 0 6px}
.MathJax_MenuClose:hover {color: white!important; border: 2px solid #CCC!important}
.MathJax_MenuClose:hover span {background-color: #CCC!important}
.MathJax_MenuClose:hover:focus {outline: none}
</style><style type="text/css">.MathJax_Preview .MJXf-math {color: inherit!important}
</style><style type="text/css">.MJX_Assistive_MathML {position: absolute!important; top: 0; left: 0; clip: rect(1px, 1px, 1px, 1px); padding: 1px 0 0 0!important; border: 0!important; height: 1px!important; width: 1px!important; overflow: hidden!important; display: block!important; -webkit-touch-callout: none; -webkit-user-select: none; -khtml-user-select: none; -moz-user-select: none; -ms-user-select: none; user-select: none}
.MJX_Assistive_MathML.MJX_Assistive_MathML_Block {width: 100%!important}
</style><style type="text/css">#MathJax_Zoom {position: absolute; background-color: #F0F0F0; overflow: auto; display: block; z-index: 301; padding: .5em; border: 1px solid black; margin: 0; font-weight: normal; font-style: normal; text-align: left; text-indent: 0; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; -webkit-box-sizing: content-box; -moz-box-sizing: content-box; box-sizing: content-box; box-shadow: 5px 5px 15px #AAAAAA; -webkit-box-shadow: 5px 5px 15px #AAAAAA; -moz-box-shadow: 5px 5px 15px #AAAAAA; -khtml-box-shadow: 5px 5px 15px #AAAAAA; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true')}
#MathJax_ZoomOverlay {position: absolute; left: 0; top: 0; z-index: 300; display: inline-block; width: 100%; height: 100%; border: 0; padding: 0; margin: 0; background-color: white; opacity: 0; filter: alpha(opacity=0)}
#MathJax_ZoomFrame {position: relative; display: inline-block; height: 0; width: 0}
#MathJax_ZoomEventTrap {position: absolute; left: 0; top: 0; z-index: 302; display: inline-block; border: 0; padding: 0; margin: 0; background-color: white; opacity: 0; filter: alpha(opacity=0)}
</style><style type="text/css">.MathJax_Preview {color: #888; display: contents}
#MathJax_Message {position: fixed; left: 1em; bottom: 1.5em; background-color: #E6E6E6; border: 1px solid #959595; margin: 0px; padding: 2px 8px; z-index: 102; color: black; font-size: 80%; width: auto; white-space: nowrap}
#MathJax_MSIE_Frame {position: absolute; top: 0; left: 0; width: 0px; z-index: 101; border: 0px; margin: 0px; padding: 0px}
.MathJax_Error {color: #CC0000; font-style: italic}
</style><script type="text/javascript" charset="utf-8" async="" data-requirecontext="_" data-requiremodule="custom/custom" src="./elandhiya_files/custom.js.download"></script><style type="text/css">div.MathJax_MathML {text-align: center; margin: .75em 0px; display: block!important}
.MathJax_MathML {font-style: normal; font-weight: normal; line-height: normal; font-size: 100%; font-size-adjust: none; text-indent: 0; text-align: left; text-transform: none; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; direction: ltr; max-width: none; max-height: none; min-width: 0; min-height: 0; border: 0; padding: 0; margin: 0}
span.MathJax_MathML {display: inline!important}
.MathJax_mmlExBox {display: block!important; overflow: hidden; height: 1px; width: 60ex; min-height: 0; max-height: none; padding: 0; border: 0; margin: 0}
[class="MJX-tex-oldstyle"] {font-family: MathJax_Caligraphic, MathJax_Caligraphic-WEB}
[class="MJX-tex-oldstyle-bold"] {font-family: MathJax_Caligraphic, MathJax_Caligraphic-WEB; font-weight: bold}
[class="MJX-tex-caligraphic"] {font-family: MathJax_Caligraphic, MathJax_Caligraphic-WEB}
[class="MJX-tex-caligraphic-bold"] {font-family: MathJax_Caligraphic, MathJax_Caligraphic-WEB; font-weight: bold}
@font-face /*1*/ {font-family: MathJax_Caligraphic-WEB; src: url('http://localhost:8888/static/components/MathJax/fonts/HTML-CSS/TeX/otf/MathJax_Caligraphic-Regular.otf')}
@font-face /*2*/ {font-family: MathJax_Caligraphic-WEB; font-weight: bold; src: url('http://localhost:8888/static/components/MathJax/fonts/HTML-CSS/TeX/otf/MathJax_Caligraphic-Bold.otf')}
[mathvariant="double-struck"] {font-family: MathJax_AMS, MathJax_AMS-WEB}
[mathvariant="script"] {font-family: MathJax_Script, MathJax_Script-WEB}
[mathvariant="fraktur"] {font-family: MathJax_Fraktur, MathJax_Fraktur-WEB}
[mathvariant="bold-script"] {font-family: MathJax_Script, MathJax_Caligraphic-WEB; font-weight: bold}
[mathvariant="bold-fraktur"] {font-family: MathJax_Fraktur, MathJax_Fraktur-WEB; font-weight: bold}
[mathvariant="monospace"] {font-family: monospace}
[mathvariant="sans-serif"] {font-family: sans-serif}
[mathvariant="bold-sans-serif"] {font-family: sans-serif; font-weight: bold}
[mathvariant="sans-serif-italic"] {font-family: sans-serif; font-style: italic}
[mathvariant="sans-serif-bold-italic"] {font-family: sans-serif; font-style: italic; font-weight: bold}
@font-face /*3*/ {font-family: MathJax_AMS-WEB; src: url('http://localhost:8888/static/components/MathJax/fonts/HTML-CSS/TeX/otf/MathJax_AMS-Regular.otf')}
@font-face /*4*/ {font-family: MathJax_Script-WEB; src: url('http://localhost:8888/static/components/MathJax/fonts/HTML-CSS/TeX/otf/MathJax_Script-Regular.otf')}
@font-face /*5*/ {font-family: MathJax_Fraktur-WEB; src: url('http://localhost:8888/static/components/MathJax/fonts/HTML-CSS/TeX/otf/MathJax_Fraktur-Regular.otf')}
@font-face /*6*/ {font-family: MathJax_Fraktur-WEB; font-weight: bold; src: url('http://localhost:8888/static/components/MathJax/fonts/HTML-CSS/TeX/otf/MathJax_Fraktur-Bold.otf')}
</style><style type="text/css">.MJXp-script {font-size: .8em}
.MJXp-right {-webkit-transform-origin: right; -moz-transform-origin: right; -ms-transform-origin: right; -o-transform-origin: right; transform-origin: right}
.MJXp-bold {font-weight: bold}
.MJXp-italic {font-style: italic}
.MJXp-scr {font-family: MathJax_Script,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-frak {font-family: MathJax_Fraktur,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-sf {font-family: MathJax_SansSerif,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-cal {font-family: MathJax_Caligraphic,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-mono {font-family: MathJax_Typewriter,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-largeop {font-size: 150%}
.MJXp-largeop.MJXp-int {vertical-align: -.2em}
.MJXp-math {display: inline-block; line-height: 1.2; text-indent: 0; font-family: 'Times New Roman',Times,STIXGeneral,serif; white-space: nowrap; border-collapse: collapse}
.MJXp-display {display: block; text-align: center; margin: 1em 0}
.MJXp-math span {display: inline-block}
.MJXp-box {display: block!important; text-align: center}
.MJXp-box:after {content: " "}
.MJXp-rule {display: block!important; margin-top: .1em}
.MJXp-char {display: block!important}
.MJXp-mo {margin: 0 .15em}
.MJXp-mfrac {margin: 0 .125em; vertical-align: .25em}
.MJXp-denom {display: inline-table!important; width: 100%}
.MJXp-denom > * {display: table-row!important}
.MJXp-surd {vertical-align: top}
.MJXp-surd > * {display: block!important}
.MJXp-script-box > *  {display: table!important; height: 50%}
.MJXp-script-box > * > * {display: table-cell!important; vertical-align: top}
.MJXp-script-box > *:last-child > * {vertical-align: bottom}
.MJXp-script-box > * > * > * {display: block!important}
.MJXp-mphantom {visibility: hidden}
.MJXp-munderover, .MJXp-munder {display: inline-table!important}
.MJXp-over {display: inline-block!important; text-align: center}
.MJXp-over > * {display: block!important}
.MJXp-munderover > *, .MJXp-munder > * {display: table-row!important}
.MJXp-mtable {vertical-align: .25em; margin: 0 .125em}
.MJXp-mtable > * {display: inline-table!important; vertical-align: middle}
.MJXp-mtr {display: table-row!important}
.MJXp-mtd {display: table-cell!important; text-align: center; padding: .5em 0 0 .5em}
.MJXp-mtr > .MJXp-mtd:first-child {padding-left: 0}
.MJXp-mtr:first-child > .MJXp-mtd {padding-top: 0}
.MJXp-mlabeledtr {display: table-row!important}
.MJXp-mlabeledtr > .MJXp-mtd:first-child {padding-left: 0}
.MJXp-mlabeledtr:first-child > .MJXp-mtd {padding-top: 0}
.MJXp-merror {background-color: #FFFF88; color: #CC0000; border: 1px solid #CC0000; padding: 1px 3px; font-style: normal; font-size: 90%}
.MJXp-scale0 {-webkit-transform: scaleX(.0); -moz-transform: scaleX(.0); -ms-transform: scaleX(.0); -o-transform: scaleX(.0); transform: scaleX(.0)}
.MJXp-scale1 {-webkit-transform: scaleX(.1); -moz-transform: scaleX(.1); -ms-transform: scaleX(.1); -o-transform: scaleX(.1); transform: scaleX(.1)}
.MJXp-scale2 {-webkit-transform: scaleX(.2); -moz-transform: scaleX(.2); -ms-transform: scaleX(.2); -o-transform: scaleX(.2); transform: scaleX(.2)}
.MJXp-scale3 {-webkit-transform: scaleX(.3); -moz-transform: scaleX(.3); -ms-transform: scaleX(.3); -o-transform: scaleX(.3); transform: scaleX(.3)}
.MJXp-scale4 {-webkit-transform: scaleX(.4); -moz-transform: scaleX(.4); -ms-transform: scaleX(.4); -o-transform: scaleX(.4); transform: scaleX(.4)}
.MJXp-scale5 {-webkit-transform: scaleX(.5); -moz-transform: scaleX(.5); -ms-transform: scaleX(.5); -o-transform: scaleX(.5); transform: scaleX(.5)}
.MJXp-scale6 {-webkit-transform: scaleX(.6); -moz-transform: scaleX(.6); -ms-transform: scaleX(.6); -o-transform: scaleX(.6); transform: scaleX(.6)}
.MJXp-scale7 {-webkit-transform: scaleX(.7); -moz-transform: scaleX(.7); -ms-transform: scaleX(.7); -o-transform: scaleX(.7); transform: scaleX(.7)}
.MJXp-scale8 {-webkit-transform: scaleX(.8); -moz-transform: scaleX(.8); -ms-transform: scaleX(.8); -o-transform: scaleX(.8); transform: scaleX(.8)}
.MJXp-scale9 {-webkit-transform: scaleX(.9); -moz-transform: scaleX(.9); -ms-transform: scaleX(.9); -o-transform: scaleX(.9); transform: scaleX(.9)}
.MathJax_PHTML .noError {vertical-align: ; font-size: 90%; text-align: left; color: black; padding: 1px 3px; border: 1px solid}
</style><script type="text/javascript" charset="utf-8" async="" data-requirecontext="_" data-requiremodule="nbextensions/jupyter-js-widgets/extension" src="./elandhiya_files/extension.js.download"></script><style type="text/css">.MathJax_Display {text-align: center; margin: 0; position: relative; display: block!important; text-indent: 0; max-width: none; max-height: none; min-width: 0; min-height: 0; width: 100%}
.MathJax .merror {background-color: #FFFF88; color: #CC0000; border: 1px solid #CC0000; padding: 1px 3px; font-style: normal; font-size: 90%}
.MathJax .MJX-monospace {font-family: monospace}
.MathJax .MJX-sans-serif {font-family: sans-serif}
#MathJax_Tooltip {background-color: InfoBackground; color: InfoText; border: 1px solid black; box-shadow: 2px 2px 5px #AAAAAA; -webkit-box-shadow: 2px 2px 5px #AAAAAA; -moz-box-shadow: 2px 2px 5px #AAAAAA; -khtml-box-shadow: 2px 2px 5px #AAAAAA; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true'); padding: 3px 4px; z-index: 401; position: absolute; left: 0; top: 0; width: auto; height: auto; display: none}
.MathJax {display: inline; font-style: normal; font-weight: normal; line-height: normal; font-size: 100%; font-size-adjust: none; text-indent: 0; text-align: left; text-transform: none; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; direction: ltr; max-width: none; max-height: none; min-width: 0; min-height: 0; border: 0; padding: 0; margin: 0}
.MathJax:focus, body :focus .MathJax {display: inline-table}
.MathJax.MathJax_FullWidth {text-align: center; display: table-cell!important; width: 10000em!important}
.MathJax img, .MathJax nobr, .MathJax a {border: 0; padding: 0; margin: 0; max-width: none; max-height: none; min-width: 0; min-height: 0; vertical-align: 0; line-height: normal; text-decoration: none}
img.MathJax_strut {border: 0!important; padding: 0!important; margin: 0!important; vertical-align: 0!important}
.MathJax span {display: inline; position: static; border: 0; padding: 0; margin: 0; vertical-align: 0; line-height: normal; text-decoration: none; box-sizing: content-box}
.MathJax nobr {white-space: nowrap!important}
.MathJax img {display: inline!important; float: none!important}
.MathJax * {transition: none; -webkit-transition: none; -moz-transition: none; -ms-transition: none; -o-transition: none}
.MathJax_Processing {visibility: hidden; position: fixed; width: 0; height: 0; overflow: hidden}
.MathJax_Processed {display: none!important}
.MathJax_test {font-style: normal; font-weight: normal; font-size: 100%; font-size-adjust: none; text-indent: 0; text-transform: none; letter-spacing: normal; word-spacing: normal; overflow: hidden; height: 1px}
.MathJax_test.mjx-test-display {display: table!important}
.MathJax_test.mjx-test-inline {display: inline!important; margin-right: -1px}
.MathJax_test.mjx-test-default {display: block!important; clear: both}
.MathJax_ex_box {display: inline-block!important; position: absolute; overflow: hidden; min-height: 0; max-height: none; padding: 0; border: 0; margin: 0; width: 1px; height: 60ex}
.MathJax_em_box {display: inline-block!important; position: absolute; overflow: hidden; min-height: 0; max-height: none; padding: 0; border: 0; margin: 0; width: 1px; height: 60em}
.mjx-test-inline .MathJax_left_box {display: inline-block; width: 0; float: left}
.mjx-test-inline .MathJax_right_box {display: inline-block; width: 0; float: right}
.mjx-test-display .MathJax_right_box {display: table-cell!important; width: 10000em!important; min-width: 0; max-width: none; padding: 0; border: 0; margin: 0}
.MathJax .MathJax_HitBox {cursor: text; background: white; opacity: 0; filter: alpha(opacity=0)}
.MathJax .MathJax_HitBox * {filter: none; opacity: 1; background: transparent}
#MathJax_Tooltip * {filter: none; opacity: 1; background: transparent}
@font-face {font-family: MathJax_Blank; src: url('about:blank')}
.MathJax .noError {vertical-align: ; font-size: 90%; text-align: left; color: black; padding: 1px 3px; border: 1px solid}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/* Override the correction for the prompt area in https://github.com/jupyter/notebook/blob/dd41d9fd5c4f698bd7468612d877828a7eeb0e7a/IPython/html/static/notebook/less/outputarea.less#L110 */
.jupyter-widgets-output-area div.output_subarea {
  max-width: 100%;
}

/* Work-around for the bug fixed in https://github.com/jupyter/notebook/pull/2961 */
.jupyter-widgets-output-area > .out_prompt_overlay {
  display: none;
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uL3NyYy93aWRnZXRfb3V0cHV0LmNzcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7O0VBR0U7O0FBRUYsNExBQTRMO0FBQzVMO0VBQ0UsZUFBZTtBQUNqQjs7QUFFQSxtRkFBbUY7QUFDbkY7RUFDRSxhQUFhO0FBQ2YiLCJzb3VyY2VzQ29udGVudCI6WyIvKlxuICogQ29weXJpZ2h0IChjKSBKdXB5dGVyIERldmVsb3BtZW50IFRlYW0uXG4gKiBEaXN0cmlidXRlZCB1bmRlciB0aGUgdGVybXMgb2YgdGhlIE1vZGlmaWVkIEJTRCBMaWNlbnNlLlxuICovXG5cbi8qIE92ZXJyaWRlIHRoZSBjb3JyZWN0aW9uIGZvciB0aGUgcHJvbXB0IGFyZWEgaW4gaHR0cHM6Ly9naXRodWIuY29tL2p1cHl0ZXIvbm90ZWJvb2svYmxvYi9kZDQxZDlmZDVjNGY2OThiZDc0Njg2MTJkODc3ODI4YTdlZWIwZTdhL0lQeXRob24vaHRtbC9zdGF0aWMvbm90ZWJvb2svbGVzcy9vdXRwdXRhcmVhLmxlc3MjTDExMCAqL1xuLmp1cHl0ZXItd2lkZ2V0cy1vdXRwdXQtYXJlYSBkaXYub3V0cHV0X3N1YmFyZWEge1xuICBtYXgtd2lkdGg6IDEwMCU7XG59XG5cbi8qIFdvcmstYXJvdW5kIGZvciB0aGUgYnVnIGZpeGVkIGluIGh0dHBzOi8vZ2l0aHViLmNvbS9qdXB5dGVyL25vdGVib29rL3B1bGwvMjk2MSAqL1xuLmp1cHl0ZXItd2lkZ2V0cy1vdXRwdXQtYXJlYSA+IC5vdXRfcHJvbXB0X292ZXJsYXkge1xuICBkaXNwbGF5OiBub25lO1xufVxuIl0sInNvdXJjZVJvb3QiOiIifQ== */</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/* <DEPRECATED> */
.p-Widget, /* </DEPRECATED> */
.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  cursor: default;
}

/* <DEPRECATED> */
.p-Widget.p-mod-hidden, /* </DEPRECATED> */
.lm-Widget.lm-mod-hidden {
  display: none !important;
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL25vZGVfbW9kdWxlcy9AbHVtaW5vL3dpZGdldHMvc3R5bGUvd2lkZ2V0LmNzcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs4RUFPOEU7O0FBRTlFLGlCQUFpQjtBQUNqQjs7RUFFRSxzQkFBc0I7RUFDdEIsa0JBQWtCO0VBQ2xCLGdCQUFnQjtFQUNoQixlQUFlO0FBQ2pCOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSx3QkFBd0I7QUFDMUIiLCJzb3VyY2VzQ29udGVudCI6WyIvKi0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG58IENvcHlyaWdodCAoYykgSnVweXRlciBEZXZlbG9wbWVudCBUZWFtLlxufCBDb3B5cmlnaHQgKGMpIDIwMTQtMjAxNywgUGhvc3Bob3JKUyBDb250cmlidXRvcnNcbnxcbnwgRGlzdHJpYnV0ZWQgdW5kZXIgdGhlIHRlcm1zIG9mIHRoZSBCU0QgMy1DbGF1c2UgTGljZW5zZS5cbnxcbnwgVGhlIGZ1bGwgbGljZW5zZSBpcyBpbiB0aGUgZmlsZSBMSUNFTlNFLCBkaXN0cmlidXRlZCB3aXRoIHRoaXMgc29mdHdhcmUuXG58LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSovXG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtV2lkZ2V0LCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tV2lkZ2V0IHtcbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcbiAgcG9zaXRpb246IHJlbGF0aXZlO1xuICBvdmVyZmxvdzogaGlkZGVuO1xuICBjdXJzb3I6IGRlZmF1bHQ7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtV2lkZ2V0LnAtbW9kLWhpZGRlbiwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLVdpZGdldC5sbS1tb2QtaGlkZGVuIHtcbiAgZGlzcGxheTogbm9uZSAhaW1wb3J0YW50O1xufVxuIl0sInNvdXJjZVJvb3QiOiIifQ== */</style><style>.lm-AccordionPanel[data-orientation='horizontal'] > .lm-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL25vZGVfbW9kdWxlcy9AbHVtaW5vL3dpZGdldHMvc3R5bGUvYWNjb3JkaW9ucGFuZWwuY3NzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBO0VBQ0UsOERBQThEO0VBQzlELGNBQWM7RUFDZCwwQkFBMEI7RUFDMUIsMENBQTBDO0FBQzVDIiwic291cmNlc0NvbnRlbnQiOlsiLmxtLUFjY29yZGlvblBhbmVsW2RhdGEtb3JpZW50YXRpb249J2hvcml6b250YWwnXSA+IC5sbS1BY2NvcmRpb25QYW5lbC10aXRsZSB7XG4gIC8qIFRpdGxlIGlzIHJvdGF0ZWQgZm9yIGhvcml6b250YWwgYWNjb3JkaW9uIHBhbmVsIHVzaW5nIENTUyAqL1xuICBkaXNwbGF5OiBibG9jaztcbiAgdHJhbnNmb3JtLW9yaWdpbjogdG9wIGxlZnQ7XG4gIHRyYW5zZm9ybTogcm90YXRlKC05MGRlZykgdHJhbnNsYXRlKC0xMDAlKTtcbn1cbiJdLCJzb3VyY2VSb290IjoiIn0= */</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/* <DEPRECATED> */
.p-CommandPalette, /* </DEPRECATED> */
.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/* <DEPRECATED> */
.p-CommandPalette-search, /* </DEPRECATED> */
.lm-CommandPalette-search {
  flex: 0 0 auto;
}

/* <DEPRECATED> */
.p-CommandPalette-content, /* </DEPRECATED> */
.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}

/* <DEPRECATED> */
.p-CommandPalette-header, /* </DEPRECATED> */
.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

/* <DEPRECATED> */
.p-CommandPalette-item, /* </DEPRECATED> */
.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}

/* <DEPRECATED> */
.p-CommandPalette-itemIcon, /* </DEPRECATED> */
.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}

/* <DEPRECATED> */
.p-CommandPalette-itemContent, /* </DEPRECATED> */
.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}

/* <DEPRECATED> */
.p-CommandPalette-itemShortcut, /* </DEPRECATED> */
.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

/* <DEPRECATED> */
.p-CommandPalette-itemLabel, /* </DEPRECATED> */
.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-close-icon {
  border: 1px solid transparent;
  background-color: transparent;
  position: absolute;
  z-index: 1;
  right: 3%;
  top: 0;
  bottom: 0;
  margin: auto;
  padding: 7px 0;
  display: none;
  vertical-align: middle;
  outline: 0;
  cursor: pointer;
}
.lm-close-icon:after {
  content: 'X';
  display: block;
  width: 15px;
  height: 15px;
  text-align: center;
  color: #000;
  font-weight: normal;
  font-size: 12px;
  cursor: pointer;
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL25vZGVfbW9kdWxlcy9AbHVtaW5vL3dpZGdldHMvc3R5bGUvY29tbWFuZHBhbGV0dGUuY3NzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OzhFQU84RTs7QUFFOUUsaUJBQWlCO0FBQ2pCOztFQUVFLGFBQWE7RUFDYixzQkFBc0I7RUFDdEIseUJBQXlCO0VBQ3pCLHNCQUFzQjtFQUN0QixxQkFBcUI7RUFDckIsaUJBQWlCO0FBQ25COztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxjQUFjO0FBQ2hCOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxjQUFjO0VBQ2QsU0FBUztFQUNULFVBQVU7RUFDVixhQUFhO0VBQ2IsY0FBYztFQUNkLHFCQUFxQjtBQUN2Qjs7QUFFQSxpQkFBaUI7QUFDakI7O0VBRUUsZ0JBQWdCO0VBQ2hCLG1CQUFtQjtFQUNuQix1QkFBdUI7QUFDekI7O0FBRUEsaUJBQWlCO0FBQ2pCOztFQUVFLGFBQWE7RUFDYixtQkFBbUI7QUFDckI7O0FBRUEsaUJBQWlCO0FBQ2pCOztFQUVFLGNBQWM7QUFDaEI7O0FBRUEsaUJBQWlCO0FBQ2pCOztFQUVFLGNBQWM7RUFDZCxnQkFBZ0I7QUFDbEI7O0FBRUEsaUJBQWlCO0FBQ2pCOztFQUVFLGNBQWM7QUFDaEI7O0FBRUEsaUJBQWlCO0FBQ2pCOztFQUVFLGdCQUFnQjtFQUNoQixtQkFBbUI7RUFDbkIsdUJBQXVCO0FBQ3pCOztBQUVBO0VBQ0UsNkJBQTZCO0VBQzdCLDZCQUE2QjtFQUM3QixrQkFBa0I7RUFDbEIsVUFBVTtFQUNWLFNBQVM7RUFDVCxNQUFNO0VBQ04sU0FBUztFQUNULFlBQVk7RUFDWixjQUFjO0VBQ2QsYUFBYTtFQUNiLHNCQUFzQjtFQUN0QixVQUFVO0VBQ1YsZUFBZTtBQUNqQjtBQUNBO0VBQ0UsWUFBWTtFQUNaLGNBQWM7RUFDZCxXQUFXO0VBQ1gsWUFBWTtFQUNaLGtCQUFrQjtFQUNsQixXQUFXO0VBQ1gsbUJBQW1CO0VBQ25CLGVBQWU7RUFDZixlQUFlO0FBQ2pCIiwic291cmNlc0NvbnRlbnQiOlsiLyotLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxufCBDb3B5cmlnaHQgKGMpIEp1cHl0ZXIgRGV2ZWxvcG1lbnQgVGVhbS5cbnwgQ29weXJpZ2h0IChjKSAyMDE0LTIwMTcsIFBob3NwaG9ySlMgQ29udHJpYnV0b3JzXG58XG58IERpc3RyaWJ1dGVkIHVuZGVyIHRoZSB0ZXJtcyBvZiB0aGUgQlNEIDMtQ2xhdXNlIExpY2Vuc2UuXG58XG58IFRoZSBmdWxsIGxpY2Vuc2UgaXMgaW4gdGhlIGZpbGUgTElDRU5TRSwgZGlzdHJpYnV0ZWQgd2l0aCB0aGlzIHNvZnR3YXJlLlxufC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0qL1xuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLUNvbW1hbmRQYWxldHRlLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tQ29tbWFuZFBhbGV0dGUge1xuICBkaXNwbGF5OiBmbGV4O1xuICBmbGV4LWRpcmVjdGlvbjogY29sdW1uO1xuICAtd2Via2l0LXVzZXItc2VsZWN0OiBub25lO1xuICAtbW96LXVzZXItc2VsZWN0OiBub25lO1xuICAtbXMtdXNlci1zZWxlY3Q6IG5vbmU7XG4gIHVzZXItc2VsZWN0OiBub25lO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLUNvbW1hbmRQYWxldHRlLXNlYXJjaCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLUNvbW1hbmRQYWxldHRlLXNlYXJjaCB7XG4gIGZsZXg6IDAgMCBhdXRvO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLUNvbW1hbmRQYWxldHRlLWNvbnRlbnQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1Db21tYW5kUGFsZXR0ZS1jb250ZW50IHtcbiAgZmxleDogMSAxIGF1dG87XG4gIG1hcmdpbjogMDtcbiAgcGFkZGluZzogMDtcbiAgbWluLWhlaWdodDogMDtcbiAgb3ZlcmZsb3c6IGF1dG87XG4gIGxpc3Qtc3R5bGUtdHlwZTogbm9uZTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1Db21tYW5kUGFsZXR0ZS1oZWFkZXIsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1Db21tYW5kUGFsZXR0ZS1oZWFkZXIge1xuICBvdmVyZmxvdzogaGlkZGVuO1xuICB3aGl0ZS1zcGFjZTogbm93cmFwO1xuICB0ZXh0LW92ZXJmbG93OiBlbGxpcHNpcztcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1Db21tYW5kUGFsZXR0ZS1pdGVtLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tQ29tbWFuZFBhbGV0dGUtaXRlbSB7XG4gIGRpc3BsYXk6IGZsZXg7XG4gIGZsZXgtZGlyZWN0aW9uOiByb3c7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtQ29tbWFuZFBhbGV0dGUtaXRlbUljb24sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1Db21tYW5kUGFsZXR0ZS1pdGVtSWNvbiB7XG4gIGZsZXg6IDAgMCBhdXRvO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLUNvbW1hbmRQYWxldHRlLWl0ZW1Db250ZW50LCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tQ29tbWFuZFBhbGV0dGUtaXRlbUNvbnRlbnQge1xuICBmbGV4OiAxIDEgYXV0bztcbiAgb3ZlcmZsb3c6IGhpZGRlbjtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1Db21tYW5kUGFsZXR0ZS1pdGVtU2hvcnRjdXQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1Db21tYW5kUGFsZXR0ZS1pdGVtU2hvcnRjdXQge1xuICBmbGV4OiAwIDAgYXV0bztcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1Db21tYW5kUGFsZXR0ZS1pdGVtTGFiZWwsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1Db21tYW5kUGFsZXR0ZS1pdGVtTGFiZWwge1xuICBvdmVyZmxvdzogaGlkZGVuO1xuICB3aGl0ZS1zcGFjZTogbm93cmFwO1xuICB0ZXh0LW92ZXJmbG93OiBlbGxpcHNpcztcbn1cblxuLmxtLWNsb3NlLWljb24ge1xuICBib3JkZXI6IDFweCBzb2xpZCB0cmFuc3BhcmVudDtcbiAgYmFja2dyb3VuZC1jb2xvcjogdHJhbnNwYXJlbnQ7XG4gIHBvc2l0aW9uOiBhYnNvbHV0ZTtcbiAgei1pbmRleDogMTtcbiAgcmlnaHQ6IDMlO1xuICB0b3A6IDA7XG4gIGJvdHRvbTogMDtcbiAgbWFyZ2luOiBhdXRvO1xuICBwYWRkaW5nOiA3cHggMDtcbiAgZGlzcGxheTogbm9uZTtcbiAgdmVydGljYWwtYWxpZ246IG1pZGRsZTtcbiAgb3V0bGluZTogMDtcbiAgY3Vyc29yOiBwb2ludGVyO1xufVxuLmxtLWNsb3NlLWljb246YWZ0ZXIge1xuICBjb250ZW50OiAnWCc7XG4gIGRpc3BsYXk6IGJsb2NrO1xuICB3aWR0aDogMTVweDtcbiAgaGVpZ2h0OiAxNXB4O1xuICB0ZXh0LWFsaWduOiBjZW50ZXI7XG4gIGNvbG9yOiAjMDAwO1xuICBmb250LXdlaWdodDogbm9ybWFsO1xuICBmb250LXNpemU6IDEycHg7XG4gIGN1cnNvcjogcG9pbnRlcjtcbn1cbiJdLCJzb3VyY2VSb290IjoiIn0= */</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/* <DEPRECATED> */
.p-DockPanel, /* </DEPRECATED> */
.lm-DockPanel {
  z-index: 0;
}

/* <DEPRECATED> */
.p-DockPanel-widget, /* </DEPRECATED> */
.lm-DockPanel-widget {
  z-index: 0;
}

/* <DEPRECATED> */
.p-DockPanel-tabBar, /* </DEPRECATED> */
.lm-DockPanel-tabBar {
  z-index: 1;
}

/* <DEPRECATED> */
.p-DockPanel-handle, /* </DEPRECATED> */
.lm-DockPanel-handle {
  z-index: 2;
}

/* <DEPRECATED> */
.p-DockPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}

/* <DEPRECATED> */
.p-DockPanel-handle:after, /* </DEPRECATED> */
.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}

/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}

/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/* <DEPRECATED> */
.p-DockPanel-overlay, /* </DEPRECATED> */
.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}

/* <DEPRECATED> */
.p-DockPanel-overlay.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL25vZGVfbW9kdWxlcy9AbHVtaW5vL3dpZGdldHMvc3R5bGUvZG9ja3BhbmVsLmNzcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs4RUFPOEU7O0FBRTlFLGlCQUFpQjtBQUNqQjs7RUFFRSxVQUFVO0FBQ1o7O0FBRUEsaUJBQWlCO0FBQ2pCOztFQUVFLFVBQVU7QUFDWjs7QUFFQSxpQkFBaUI7QUFDakI7O0VBRUUsVUFBVTtBQUNaOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxVQUFVO0FBQ1o7O0FBRUEsaUJBQWlCO0FBQ2pCOztFQUVFLHdCQUF3QjtBQUMxQjs7QUFFQSxpQkFBaUI7QUFDakI7O0VBRUUsa0JBQWtCO0VBQ2xCLE1BQU07RUFDTixPQUFPO0VBQ1AsV0FBVztFQUNYLFlBQVk7RUFDWixXQUFXO0FBQ2I7O0FBRUEsaUJBQWlCO0FBQ2pCOzs7RUFHRSxpQkFBaUI7QUFDbkI7O0FBRUEsaUJBQWlCO0FBQ2pCOzs7RUFHRSxpQkFBaUI7QUFDbkI7O0FBRUEsaUJBQWlCO0FBQ2pCOzs7RUFHRSxTQUFTO0VBQ1QsY0FBYztFQUNkLDJCQUEyQjtBQUM3Qjs7QUFFQSxpQkFBaUI7QUFDakI7OztFQUdFLFFBQVE7RUFDUixlQUFlO0VBQ2YsMkJBQTJCO0FBQzdCOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxVQUFVO0VBQ1Ysc0JBQXNCO0VBQ3RCLG9CQUFvQjtBQUN0Qjs7QUFFQSxpQkFBaUI7QUFDakI7O0VBRUUsd0JBQXdCO0FBQzFCIiwic291cmNlc0NvbnRlbnQiOlsiLyotLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxufCBDb3B5cmlnaHQgKGMpIEp1cHl0ZXIgRGV2ZWxvcG1lbnQgVGVhbS5cbnwgQ29weXJpZ2h0IChjKSAyMDE0LTIwMTcsIFBob3NwaG9ySlMgQ29udHJpYnV0b3JzXG58XG58IERpc3RyaWJ1dGVkIHVuZGVyIHRoZSB0ZXJtcyBvZiB0aGUgQlNEIDMtQ2xhdXNlIExpY2Vuc2UuXG58XG58IFRoZSBmdWxsIGxpY2Vuc2UgaXMgaW4gdGhlIGZpbGUgTElDRU5TRSwgZGlzdHJpYnV0ZWQgd2l0aCB0aGlzIHNvZnR3YXJlLlxufC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0qL1xuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLURvY2tQYW5lbCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLURvY2tQYW5lbCB7XG4gIHotaW5kZXg6IDA7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtRG9ja1BhbmVsLXdpZGdldCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLURvY2tQYW5lbC13aWRnZXQge1xuICB6LWluZGV4OiAwO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLURvY2tQYW5lbC10YWJCYXIsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1Eb2NrUGFuZWwtdGFiQmFyIHtcbiAgei1pbmRleDogMTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1Eb2NrUGFuZWwtaGFuZGxlLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tRG9ja1BhbmVsLWhhbmRsZSB7XG4gIHotaW5kZXg6IDI7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtRG9ja1BhbmVsLWhhbmRsZS5wLW1vZC1oaWRkZW4sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1Eb2NrUGFuZWwtaGFuZGxlLmxtLW1vZC1oaWRkZW4ge1xuICBkaXNwbGF5OiBub25lICFpbXBvcnRhbnQ7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtRG9ja1BhbmVsLWhhbmRsZTphZnRlciwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLURvY2tQYW5lbC1oYW5kbGU6YWZ0ZXIge1xuICBwb3NpdGlvbjogYWJzb2x1dGU7XG4gIHRvcDogMDtcbiAgbGVmdDogMDtcbiAgd2lkdGg6IDEwMCU7XG4gIGhlaWdodDogMTAwJTtcbiAgY29udGVudDogJyc7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtRG9ja1BhbmVsLWhhbmRsZVtkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ10sXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tRG9ja1BhbmVsLWhhbmRsZVtkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ10ge1xuICBjdXJzb3I6IGV3LXJlc2l6ZTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1Eb2NrUGFuZWwtaGFuZGxlW2RhdGEtb3JpZW50YXRpb249J3ZlcnRpY2FsJ10sXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tRG9ja1BhbmVsLWhhbmRsZVtkYXRhLW9yaWVudGF0aW9uPSd2ZXJ0aWNhbCddIHtcbiAgY3Vyc29yOiBucy1yZXNpemU7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtRG9ja1BhbmVsLWhhbmRsZVtkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ106YWZ0ZXIsXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tRG9ja1BhbmVsLWhhbmRsZVtkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ106YWZ0ZXIge1xuICBsZWZ0OiA1MCU7XG4gIG1pbi13aWR0aDogOHB4O1xuICB0cmFuc2Zvcm06IHRyYW5zbGF0ZVgoLTUwJSk7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtRG9ja1BhbmVsLWhhbmRsZVtkYXRhLW9yaWVudGF0aW9uPSd2ZXJ0aWNhbCddOmFmdGVyLFxuLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLURvY2tQYW5lbC1oYW5kbGVbZGF0YS1vcmllbnRhdGlvbj0ndmVydGljYWwnXTphZnRlciB7XG4gIHRvcDogNTAlO1xuICBtaW4taGVpZ2h0OiA4cHg7XG4gIHRyYW5zZm9ybTogdHJhbnNsYXRlWSgtNTAlKTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1Eb2NrUGFuZWwtb3ZlcmxheSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLURvY2tQYW5lbC1vdmVybGF5IHtcbiAgei1pbmRleDogMztcbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcbiAgcG9pbnRlci1ldmVudHM6IG5vbmU7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtRG9ja1BhbmVsLW92ZXJsYXkucC1tb2QtaGlkZGVuLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tRG9ja1BhbmVsLW92ZXJsYXkubG0tbW9kLWhpZGRlbiB7XG4gIGRpc3BsYXk6IG5vbmUgIWltcG9ydGFudDtcbn1cbiJdLCJzb3VyY2VSb290IjoiIn0= */</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/* <DEPRECATED> */
.p-Menu, /* </DEPRECATED> */
.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/* <DEPRECATED> */
.p-Menu-content, /* </DEPRECATED> */
.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}

/* <DEPRECATED> */
.p-Menu-item, /* </DEPRECATED> */
.lm-Menu-item {
  display: table-row;
}

/* <DEPRECATED> */
.p-Menu-item.p-mod-hidden,
.p-Menu-item.p-mod-collapsed,
/* </DEPRECATED> */
.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}

/* <DEPRECATED> */
.p-Menu-itemIcon,
.p-Menu-itemSubmenuIcon,
/* </DEPRECATED> */
.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}

/* <DEPRECATED> */
.p-Menu-itemLabel, /* </DEPRECATED> */
.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}

/* <DEPRECATED> */
.p-Menu-itemShortcut, /* </DEPRECATED> */
.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL25vZGVfbW9kdWxlcy9AbHVtaW5vL3dpZGdldHMvc3R5bGUvbWVudS5jc3MiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7OEVBTzhFOztBQUU5RSxpQkFBaUI7QUFDakI7O0VBRUUsY0FBYztFQUNkLGtCQUFrQjtFQUNsQixtQkFBbUI7RUFDbkIsa0JBQWtCO0VBQ2xCLGdCQUFnQjtFQUNoQixhQUFhO0VBQ2IseUJBQXlCO0VBQ3pCLHNCQUFzQjtFQUN0QixxQkFBcUI7RUFDckIsaUJBQWlCO0FBQ25COztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxTQUFTO0VBQ1QsVUFBVTtFQUNWLGNBQWM7RUFDZCxxQkFBcUI7QUFDdkI7O0FBRUEsaUJBQWlCO0FBQ2pCOztFQUVFLGtCQUFrQjtBQUNwQjs7QUFFQSxpQkFBaUI7QUFDakI7Ozs7O0VBS0Usd0JBQXdCO0FBQzFCOztBQUVBLGlCQUFpQjtBQUNqQjs7Ozs7RUFLRSxtQkFBbUI7RUFDbkIsa0JBQWtCO0FBQ3BCOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxtQkFBbUI7RUFDbkIsZ0JBQWdCO0FBQ2xCOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxtQkFBbUI7RUFDbkIsaUJBQWlCO0FBQ25CIiwic291cmNlc0NvbnRlbnQiOlsiLyotLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxufCBDb3B5cmlnaHQgKGMpIEp1cHl0ZXIgRGV2ZWxvcG1lbnQgVGVhbS5cbnwgQ29weXJpZ2h0IChjKSAyMDE0LTIwMTcsIFBob3NwaG9ySlMgQ29udHJpYnV0b3JzXG58XG58IERpc3RyaWJ1dGVkIHVuZGVyIHRoZSB0ZXJtcyBvZiB0aGUgQlNEIDMtQ2xhdXNlIExpY2Vuc2UuXG58XG58IFRoZSBmdWxsIGxpY2Vuc2UgaXMgaW4gdGhlIGZpbGUgTElDRU5TRSwgZGlzdHJpYnV0ZWQgd2l0aCB0aGlzIHNvZnR3YXJlLlxufC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0qL1xuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLU1lbnUsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1NZW51IHtcbiAgei1pbmRleDogMTAwMDA7XG4gIHBvc2l0aW9uOiBhYnNvbHV0ZTtcbiAgd2hpdGUtc3BhY2U6IG5vd3JhcDtcbiAgb3ZlcmZsb3cteDogaGlkZGVuO1xuICBvdmVyZmxvdy15OiBhdXRvO1xuICBvdXRsaW5lOiBub25lO1xuICAtd2Via2l0LXVzZXItc2VsZWN0OiBub25lO1xuICAtbW96LXVzZXItc2VsZWN0OiBub25lO1xuICAtbXMtdXNlci1zZWxlY3Q6IG5vbmU7XG4gIHVzZXItc2VsZWN0OiBub25lO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLU1lbnUtY29udGVudCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLU1lbnUtY29udGVudCB7XG4gIG1hcmdpbjogMDtcbiAgcGFkZGluZzogMDtcbiAgZGlzcGxheTogdGFibGU7XG4gIGxpc3Qtc3R5bGUtdHlwZTogbm9uZTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1NZW51LWl0ZW0sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1NZW51LWl0ZW0ge1xuICBkaXNwbGF5OiB0YWJsZS1yb3c7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtTWVudS1pdGVtLnAtbW9kLWhpZGRlbixcbi5wLU1lbnUtaXRlbS5wLW1vZC1jb2xsYXBzZWQsXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tTWVudS1pdGVtLmxtLW1vZC1oaWRkZW4sXG4ubG0tTWVudS1pdGVtLmxtLW1vZC1jb2xsYXBzZWQge1xuICBkaXNwbGF5OiBub25lICFpbXBvcnRhbnQ7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtTWVudS1pdGVtSWNvbixcbi5wLU1lbnUtaXRlbVN1Ym1lbnVJY29uLFxuLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLU1lbnUtaXRlbUljb24sXG4ubG0tTWVudS1pdGVtU3VibWVudUljb24ge1xuICBkaXNwbGF5OiB0YWJsZS1jZWxsO1xuICB0ZXh0LWFsaWduOiBjZW50ZXI7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtTWVudS1pdGVtTGFiZWwsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1NZW51LWl0ZW1MYWJlbCB7XG4gIGRpc3BsYXk6IHRhYmxlLWNlbGw7XG4gIHRleHQtYWxpZ246IGxlZnQ7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtTWVudS1pdGVtU2hvcnRjdXQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1NZW51LWl0ZW1TaG9ydGN1dCB7XG4gIGRpc3BsYXk6IHRhYmxlLWNlbGw7XG4gIHRleHQtYWxpZ246IHJpZ2h0O1xufVxuIl0sInNvdXJjZVJvb3QiOiIifQ== */</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/* <DEPRECATED> */
.p-MenuBar, /* </DEPRECATED> */
.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/* <DEPRECATED> */
.p-MenuBar-content, /* </DEPRECATED> */
.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}

/* <DEPRECATED> */
.p--MenuBar-item, /* </DEPRECATED> */
.lm-MenuBar-item {
  box-sizing: border-box;
}

/* <DEPRECATED> */
.p-MenuBar-itemIcon,
.p-MenuBar-itemLabel,
/* </DEPRECATED> */
.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL25vZGVfbW9kdWxlcy9AbHVtaW5vL3dpZGdldHMvc3R5bGUvbWVudWJhci5jc3MiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7OEVBTzhFOztBQUU5RSxpQkFBaUI7QUFDakI7O0VBRUUsYUFBYTtFQUNiLHlCQUF5QjtFQUN6QixzQkFBc0I7RUFDdEIscUJBQXFCO0VBQ3JCLGlCQUFpQjtBQUNuQjs7QUFFQSxpQkFBaUI7QUFDakI7O0VBRUUsU0FBUztFQUNULFVBQVU7RUFDVixhQUFhO0VBQ2IsbUJBQW1CO0VBQ25CLHFCQUFxQjtBQUN2Qjs7QUFFQSxpQkFBaUI7QUFDakI7O0VBRUUsc0JBQXNCO0FBQ3hCOztBQUVBLGlCQUFpQjtBQUNqQjs7Ozs7RUFLRSxxQkFBcUI7QUFDdkIiLCJzb3VyY2VzQ29udGVudCI6WyIvKi0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG58IENvcHlyaWdodCAoYykgSnVweXRlciBEZXZlbG9wbWVudCBUZWFtLlxufCBDb3B5cmlnaHQgKGMpIDIwMTQtMjAxNywgUGhvc3Bob3JKUyBDb250cmlidXRvcnNcbnxcbnwgRGlzdHJpYnV0ZWQgdW5kZXIgdGhlIHRlcm1zIG9mIHRoZSBCU0QgMy1DbGF1c2UgTGljZW5zZS5cbnxcbnwgVGhlIGZ1bGwgbGljZW5zZSBpcyBpbiB0aGUgZmlsZSBMSUNFTlNFLCBkaXN0cmlidXRlZCB3aXRoIHRoaXMgc29mdHdhcmUuXG58LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSovXG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtTWVudUJhciwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLU1lbnVCYXIge1xuICBvdXRsaW5lOiBub25lO1xuICAtd2Via2l0LXVzZXItc2VsZWN0OiBub25lO1xuICAtbW96LXVzZXItc2VsZWN0OiBub25lO1xuICAtbXMtdXNlci1zZWxlY3Q6IG5vbmU7XG4gIHVzZXItc2VsZWN0OiBub25lO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLU1lbnVCYXItY29udGVudCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLU1lbnVCYXItY29udGVudCB7XG4gIG1hcmdpbjogMDtcbiAgcGFkZGluZzogMDtcbiAgZGlzcGxheTogZmxleDtcbiAgZmxleC1kaXJlY3Rpb246IHJvdztcbiAgbGlzdC1zdHlsZS10eXBlOiBub25lO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLS1NZW51QmFyLWl0ZW0sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1NZW51QmFyLWl0ZW0ge1xuICBib3gtc2l6aW5nOiBib3JkZXItYm94O1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLU1lbnVCYXItaXRlbUljb24sXG4ucC1NZW51QmFyLWl0ZW1MYWJlbCxcbi8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1NZW51QmFyLWl0ZW1JY29uLFxuLmxtLU1lbnVCYXItaXRlbUxhYmVsIHtcbiAgZGlzcGxheTogaW5saW5lLWJsb2NrO1xufVxuIl0sInNvdXJjZVJvb3QiOiIifQ== */</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/* <DEPRECATED> */
.p-ScrollBar, /* </DEPRECATED> */
.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/* <DEPRECATED> */
.p-ScrollBar[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}

/* <DEPRECATED> */
.p-ScrollBar[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}

/* <DEPRECATED> */
.p-ScrollBar-button, /* </DEPRECATED> */
.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}

/* <DEPRECATED> */
.p-ScrollBar-track, /* </DEPRECATED> */
.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}

/* <DEPRECATED> */
.p-ScrollBar-thumb, /* </DEPRECATED> */
.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL25vZGVfbW9kdWxlcy9AbHVtaW5vL3dpZGdldHMvc3R5bGUvc2Nyb2xsYmFyLmNzcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs4RUFPOEU7O0FBRTlFLGlCQUFpQjtBQUNqQjs7RUFFRSxhQUFhO0VBQ2IseUJBQXlCO0VBQ3pCLHNCQUFzQjtFQUN0QixxQkFBcUI7RUFDckIsaUJBQWlCO0FBQ25COztBQUVBLGlCQUFpQjtBQUNqQjs7O0VBR0UsbUJBQW1CO0FBQ3JCOztBQUVBLGlCQUFpQjtBQUNqQjs7O0VBR0Usc0JBQXNCO0FBQ3hCOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxzQkFBc0I7RUFDdEIsY0FBYztBQUNoQjs7QUFFQSxpQkFBaUI7QUFDakI7O0VBRUUsc0JBQXNCO0VBQ3RCLGtCQUFrQjtFQUNsQixnQkFBZ0I7RUFDaEIsY0FBYztBQUNoQjs7QUFFQSxpQkFBaUI7QUFDakI7O0VBRUUsc0JBQXNCO0VBQ3RCLGtCQUFrQjtBQUNwQiIsInNvdXJjZXNDb250ZW50IjpbIi8qLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbnwgQ29weXJpZ2h0IChjKSBKdXB5dGVyIERldmVsb3BtZW50IFRlYW0uXG58IENvcHlyaWdodCAoYykgMjAxNC0yMDE3LCBQaG9zcGhvckpTIENvbnRyaWJ1dG9yc1xufFxufCBEaXN0cmlidXRlZCB1bmRlciB0aGUgdGVybXMgb2YgdGhlIEJTRCAzLUNsYXVzZSBMaWNlbnNlLlxufFxufCBUaGUgZnVsbCBsaWNlbnNlIGlzIGluIHRoZSBmaWxlIExJQ0VOU0UsIGRpc3RyaWJ1dGVkIHdpdGggdGhpcyBzb2Z0d2FyZS5cbnwtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tKi9cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1TY3JvbGxCYXIsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1TY3JvbGxCYXIge1xuICBkaXNwbGF5OiBmbGV4O1xuICAtd2Via2l0LXVzZXItc2VsZWN0OiBub25lO1xuICAtbW96LXVzZXItc2VsZWN0OiBub25lO1xuICAtbXMtdXNlci1zZWxlY3Q6IG5vbmU7XG4gIHVzZXItc2VsZWN0OiBub25lO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLVNjcm9sbEJhcltkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ10sXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tU2Nyb2xsQmFyW2RhdGEtb3JpZW50YXRpb249J2hvcml6b250YWwnXSB7XG4gIGZsZXgtZGlyZWN0aW9uOiByb3c7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtU2Nyb2xsQmFyW2RhdGEtb3JpZW50YXRpb249J3ZlcnRpY2FsJ10sXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tU2Nyb2xsQmFyW2RhdGEtb3JpZW50YXRpb249J3ZlcnRpY2FsJ10ge1xuICBmbGV4LWRpcmVjdGlvbjogY29sdW1uO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLVNjcm9sbEJhci1idXR0b24sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1TY3JvbGxCYXItYnV0dG9uIHtcbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcbiAgZmxleDogMCAwIGF1dG87XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtU2Nyb2xsQmFyLXRyYWNrLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tU2Nyb2xsQmFyLXRyYWNrIHtcbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcbiAgcG9zaXRpb246IHJlbGF0aXZlO1xuICBvdmVyZmxvdzogaGlkZGVuO1xuICBmbGV4OiAxIDEgYXV0bztcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1TY3JvbGxCYXItdGh1bWIsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1TY3JvbGxCYXItdGh1bWIge1xuICBib3gtc2l6aW5nOiBib3JkZXItYm94O1xuICBwb3NpdGlvbjogYWJzb2x1dGU7XG59XG4iXSwic291cmNlUm9vdCI6IiJ9 */</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/* <DEPRECATED> */
.p-SplitPanel-child, /* </DEPRECATED> */
.lm-SplitPanel-child {
  z-index: 0;
}

/* <DEPRECATED> */
.p-SplitPanel-handle, /* </DEPRECATED> */
.lm-SplitPanel-handle {
  z-index: 1;
}

/* <DEPRECATED> */
.p-SplitPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}

/* <DEPRECATED> */
.p-SplitPanel-handle:after, /* </DEPRECATED> */
.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}

/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}

/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL25vZGVfbW9kdWxlcy9AbHVtaW5vL3dpZGdldHMvc3R5bGUvc3BsaXRwYW5lbC5jc3MiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7OEVBTzhFOztBQUU5RSxpQkFBaUI7QUFDakI7O0VBRUUsVUFBVTtBQUNaOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxVQUFVO0FBQ1o7O0FBRUEsaUJBQWlCO0FBQ2pCOztFQUVFLHdCQUF3QjtBQUMxQjs7QUFFQSxpQkFBaUI7QUFDakI7O0VBRUUsa0JBQWtCO0VBQ2xCLE1BQU07RUFDTixPQUFPO0VBQ1AsV0FBVztFQUNYLFlBQVk7RUFDWixXQUFXO0FBQ2I7O0FBRUEsaUJBQWlCO0FBQ2pCOzs7RUFHRSxpQkFBaUI7QUFDbkI7O0FBRUEsaUJBQWlCO0FBQ2pCOzs7RUFHRSxpQkFBaUI7QUFDbkI7O0FBRUEsaUJBQWlCO0FBQ2pCOzs7RUFHRSxTQUFTO0VBQ1QsY0FBYztFQUNkLDJCQUEyQjtBQUM3Qjs7QUFFQSxpQkFBaUI7QUFDakI7OztFQUdFLFFBQVE7RUFDUixlQUFlO0VBQ2YsMkJBQTJCO0FBQzdCIiwic291cmNlc0NvbnRlbnQiOlsiLyotLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxufCBDb3B5cmlnaHQgKGMpIEp1cHl0ZXIgRGV2ZWxvcG1lbnQgVGVhbS5cbnwgQ29weXJpZ2h0IChjKSAyMDE0LTIwMTcsIFBob3NwaG9ySlMgQ29udHJpYnV0b3JzXG58XG58IERpc3RyaWJ1dGVkIHVuZGVyIHRoZSB0ZXJtcyBvZiB0aGUgQlNEIDMtQ2xhdXNlIExpY2Vuc2UuXG58XG58IFRoZSBmdWxsIGxpY2Vuc2UgaXMgaW4gdGhlIGZpbGUgTElDRU5TRSwgZGlzdHJpYnV0ZWQgd2l0aCB0aGlzIHNvZnR3YXJlLlxufC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0qL1xuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLVNwbGl0UGFuZWwtY2hpbGQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1TcGxpdFBhbmVsLWNoaWxkIHtcbiAgei1pbmRleDogMDtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1TcGxpdFBhbmVsLWhhbmRsZSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLVNwbGl0UGFuZWwtaGFuZGxlIHtcbiAgei1pbmRleDogMTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1TcGxpdFBhbmVsLWhhbmRsZS5wLW1vZC1oaWRkZW4sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1TcGxpdFBhbmVsLWhhbmRsZS5sbS1tb2QtaGlkZGVuIHtcbiAgZGlzcGxheTogbm9uZSAhaW1wb3J0YW50O1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLVNwbGl0UGFuZWwtaGFuZGxlOmFmdGVyLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tU3BsaXRQYW5lbC1oYW5kbGU6YWZ0ZXIge1xuICBwb3NpdGlvbjogYWJzb2x1dGU7XG4gIHRvcDogMDtcbiAgbGVmdDogMDtcbiAgd2lkdGg6IDEwMCU7XG4gIGhlaWdodDogMTAwJTtcbiAgY29udGVudDogJyc7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtU3BsaXRQYW5lbFtkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ10gPiAucC1TcGxpdFBhbmVsLWhhbmRsZSxcbi8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1TcGxpdFBhbmVsW2RhdGEtb3JpZW50YXRpb249J2hvcml6b250YWwnXSA+IC5sbS1TcGxpdFBhbmVsLWhhbmRsZSB7XG4gIGN1cnNvcjogZXctcmVzaXplO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLVNwbGl0UGFuZWxbZGF0YS1vcmllbnRhdGlvbj0ndmVydGljYWwnXSA+IC5wLVNwbGl0UGFuZWwtaGFuZGxlLFxuLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLVNwbGl0UGFuZWxbZGF0YS1vcmllbnRhdGlvbj0ndmVydGljYWwnXSA+IC5sbS1TcGxpdFBhbmVsLWhhbmRsZSB7XG4gIGN1cnNvcjogbnMtcmVzaXplO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLVNwbGl0UGFuZWxbZGF0YS1vcmllbnRhdGlvbj0naG9yaXpvbnRhbCddID4gLnAtU3BsaXRQYW5lbC1oYW5kbGU6YWZ0ZXIsXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tU3BsaXRQYW5lbFtkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ10gPiAubG0tU3BsaXRQYW5lbC1oYW5kbGU6YWZ0ZXIge1xuICBsZWZ0OiA1MCU7XG4gIG1pbi13aWR0aDogOHB4O1xuICB0cmFuc2Zvcm06IHRyYW5zbGF0ZVgoLTUwJSk7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtU3BsaXRQYW5lbFtkYXRhLW9yaWVudGF0aW9uPSd2ZXJ0aWNhbCddID4gLnAtU3BsaXRQYW5lbC1oYW5kbGU6YWZ0ZXIsXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tU3BsaXRQYW5lbFtkYXRhLW9yaWVudGF0aW9uPSd2ZXJ0aWNhbCddID4gLmxtLVNwbGl0UGFuZWwtaGFuZGxlOmFmdGVyIHtcbiAgdG9wOiA1MCU7XG4gIG1pbi1oZWlnaHQ6IDhweDtcbiAgdHJhbnNmb3JtOiB0cmFuc2xhdGVZKC01MCUpO1xufVxuIl0sInNvdXJjZVJvb3QiOiIifQ== */</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/* <DEPRECATED> */
.p-TabBar, /* </DEPRECATED> */
.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/* <DEPRECATED> */
.p-TabBar[data-orientation='horizontal'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
  align-items: flex-end;
}

/* <DEPRECATED> */
.p-TabBar[data-orientation='vertical'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
  align-items: flex-end;
}

/* <DEPRECATED> */
.p-TabBar-content, /* </DEPRECATED> */
.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}

/* <DEPRECATED> */
.p-TabBar[data-orientation='horizontal'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}

/* <DEPRECATED> */
.p-TabBar[data-orientation='vertical'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}

/* <DEPRECATED> */
.p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
  touch-action: none; /* Disable native Drag/Drop */
}

/* <DEPRECATED> */
.p-TabBar-tabIcon,
.p-TabBar-tabCloseIcon,
/* </DEPRECATED> */
.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}

/* <DEPRECATED> */
.p-TabBar-tabLabel, /* </DEPRECATED> */
.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}

.lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
}

/* <DEPRECATED> */
.p-TabBar-tab.p-mod-hidden, /* </DEPRECATED> */
.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar-addButton.lm-mod-hidden {
  display: none !important;
}

/* <DEPRECATED> */
.p-TabBar.p-mod-dragging .p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}

/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='horizontal'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}

/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='vertical'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}

/* <DEPRECATED> */
.p-TabBar.p-mod-dragging .p-TabBar-tab.p-mod-dragging,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

.lm-TabBar-tabLabel .lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
  background: inherit;
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL25vZGVfbW9kdWxlcy9AbHVtaW5vL3dpZGdldHMvc3R5bGUvdGFiYmFyLmNzcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs4RUFPOEU7O0FBRTlFLGlCQUFpQjtBQUNqQjs7RUFFRSxhQUFhO0VBQ2IseUJBQXlCO0VBQ3pCLHNCQUFzQjtFQUN0QixxQkFBcUI7RUFDckIsaUJBQWlCO0FBQ25COztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxtQkFBbUI7RUFDbkIscUJBQXFCO0FBQ3ZCOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxzQkFBc0I7RUFDdEIscUJBQXFCO0FBQ3ZCOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxTQUFTO0VBQ1QsVUFBVTtFQUNWLGFBQWE7RUFDYixjQUFjO0VBQ2QscUJBQXFCO0FBQ3ZCOztBQUVBLGlCQUFpQjtBQUNqQjs7O0VBR0UsbUJBQW1CO0FBQ3JCOztBQUVBLGlCQUFpQjtBQUNqQjs7O0VBR0Usc0JBQXNCO0FBQ3hCOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxhQUFhO0VBQ2IsbUJBQW1CO0VBQ25CLHNCQUFzQjtFQUN0QixnQkFBZ0I7RUFDaEIsa0JBQWtCLEVBQUUsNkJBQTZCO0FBQ25EOztBQUVBLGlCQUFpQjtBQUNqQjs7Ozs7RUFLRSxjQUFjO0FBQ2hCOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxjQUFjO0VBQ2QsZ0JBQWdCO0VBQ2hCLG1CQUFtQjtBQUNyQjs7QUFFQTtFQUNFLGdCQUFnQjtFQUNoQixXQUFXO0VBQ1gsc0JBQXNCO0FBQ3hCOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSx3QkFBd0I7QUFDMUI7O0FBRUE7RUFDRSx3QkFBd0I7QUFDMUI7O0FBRUEsaUJBQWlCO0FBQ2pCOztFQUVFLGtCQUFrQjtBQUNwQjs7QUFFQSxpQkFBaUI7QUFDakI7OztFQUdFLE9BQU87RUFDUCwyQkFBMkI7QUFDN0I7O0FBRUEsaUJBQWlCO0FBQ2pCOzs7RUFHRSxNQUFNO0VBQ04sMEJBQTBCO0FBQzVCOztBQUVBLGlCQUFpQjtBQUNqQjs7O0VBR0UsZ0JBQWdCO0FBQ2xCOztBQUVBO0VBQ0UsZ0JBQWdCO0VBQ2hCLFdBQVc7RUFDWCxzQkFBc0I7RUFDdEIsbUJBQW1CO0FBQ3JCIiwic291cmNlc0NvbnRlbnQiOlsiLyotLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxufCBDb3B5cmlnaHQgKGMpIEp1cHl0ZXIgRGV2ZWxvcG1lbnQgVGVhbS5cbnwgQ29weXJpZ2h0IChjKSAyMDE0LTIwMTcsIFBob3NwaG9ySlMgQ29udHJpYnV0b3JzXG58XG58IERpc3RyaWJ1dGVkIHVuZGVyIHRoZSB0ZXJtcyBvZiB0aGUgQlNEIDMtQ2xhdXNlIExpY2Vuc2UuXG58XG58IFRoZSBmdWxsIGxpY2Vuc2UgaXMgaW4gdGhlIGZpbGUgTElDRU5TRSwgZGlzdHJpYnV0ZWQgd2l0aCB0aGlzIHNvZnR3YXJlLlxufC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0qL1xuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLVRhYkJhciwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLVRhYkJhciB7XG4gIGRpc3BsYXk6IGZsZXg7XG4gIC13ZWJraXQtdXNlci1zZWxlY3Q6IG5vbmU7XG4gIC1tb3otdXNlci1zZWxlY3Q6IG5vbmU7XG4gIC1tcy11c2VyLXNlbGVjdDogbm9uZTtcbiAgdXNlci1zZWxlY3Q6IG5vbmU7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtVGFiQmFyW2RhdGEtb3JpZW50YXRpb249J2hvcml6b250YWwnXSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLVRhYkJhcltkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ10ge1xuICBmbGV4LWRpcmVjdGlvbjogcm93O1xuICBhbGlnbi1pdGVtczogZmxleC1lbmQ7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtVGFiQmFyW2RhdGEtb3JpZW50YXRpb249J3ZlcnRpY2FsJ10sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1UYWJCYXJbZGF0YS1vcmllbnRhdGlvbj0ndmVydGljYWwnXSB7XG4gIGZsZXgtZGlyZWN0aW9uOiBjb2x1bW47XG4gIGFsaWduLWl0ZW1zOiBmbGV4LWVuZDtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1UYWJCYXItY29udGVudCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLVRhYkJhci1jb250ZW50IHtcbiAgbWFyZ2luOiAwO1xuICBwYWRkaW5nOiAwO1xuICBkaXNwbGF5OiBmbGV4O1xuICBmbGV4OiAxIDEgYXV0bztcbiAgbGlzdC1zdHlsZS10eXBlOiBub25lO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLVRhYkJhcltkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ10gPiAucC1UYWJCYXItY29udGVudCxcbi8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1UYWJCYXJbZGF0YS1vcmllbnRhdGlvbj0naG9yaXpvbnRhbCddID4gLmxtLVRhYkJhci1jb250ZW50IHtcbiAgZmxleC1kaXJlY3Rpb246IHJvdztcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1UYWJCYXJbZGF0YS1vcmllbnRhdGlvbj0ndmVydGljYWwnXSA+IC5wLVRhYkJhci1jb250ZW50LFxuLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLVRhYkJhcltkYXRhLW9yaWVudGF0aW9uPSd2ZXJ0aWNhbCddID4gLmxtLVRhYkJhci1jb250ZW50IHtcbiAgZmxleC1kaXJlY3Rpb246IGNvbHVtbjtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1UYWJCYXItdGFiLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tVGFiQmFyLXRhYiB7XG4gIGRpc3BsYXk6IGZsZXg7XG4gIGZsZXgtZGlyZWN0aW9uOiByb3c7XG4gIGJveC1zaXppbmc6IGJvcmRlci1ib3g7XG4gIG92ZXJmbG93OiBoaWRkZW47XG4gIHRvdWNoLWFjdGlvbjogbm9uZTsgLyogRGlzYWJsZSBuYXRpdmUgRHJhZy9Ecm9wICovXG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtVGFiQmFyLXRhYkljb24sXG4ucC1UYWJCYXItdGFiQ2xvc2VJY29uLFxuLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLVRhYkJhci10YWJJY29uLFxuLmxtLVRhYkJhci10YWJDbG9zZUljb24ge1xuICBmbGV4OiAwIDAgYXV0bztcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1UYWJCYXItdGFiTGFiZWwsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1UYWJCYXItdGFiTGFiZWwge1xuICBmbGV4OiAxIDEgYXV0bztcbiAgb3ZlcmZsb3c6IGhpZGRlbjtcbiAgd2hpdGUtc3BhY2U6IG5vd3JhcDtcbn1cblxuLmxtLVRhYkJhci10YWJJbnB1dCB7XG4gIHVzZXItc2VsZWN0OiBhbGw7XG4gIHdpZHRoOiAxMDAlO1xuICBib3gtc2l6aW5nOiBib3JkZXItYm94O1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLVRhYkJhci10YWIucC1tb2QtaGlkZGVuLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tVGFiQmFyLXRhYi5sbS1tb2QtaGlkZGVuIHtcbiAgZGlzcGxheTogbm9uZSAhaW1wb3J0YW50O1xufVxuXG4ubG0tVGFiQmFyLWFkZEJ1dHRvbi5sbS1tb2QtaGlkZGVuIHtcbiAgZGlzcGxheTogbm9uZSAhaW1wb3J0YW50O1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLVRhYkJhci5wLW1vZC1kcmFnZ2luZyAucC1UYWJCYXItdGFiLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tVGFiQmFyLmxtLW1vZC1kcmFnZ2luZyAubG0tVGFiQmFyLXRhYiB7XG4gIHBvc2l0aW9uOiByZWxhdGl2ZTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ucC1UYWJCYXIucC1tb2QtZHJhZ2dpbmdbZGF0YS1vcmllbnRhdGlvbj0naG9yaXpvbnRhbCddIC5wLVRhYkJhci10YWIsXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tVGFiQmFyLmxtLW1vZC1kcmFnZ2luZ1tkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ10gLmxtLVRhYkJhci10YWIge1xuICBsZWZ0OiAwO1xuICB0cmFuc2l0aW9uOiBsZWZ0IDE1MG1zIGVhc2U7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtVGFiQmFyLnAtbW9kLWRyYWdnaW5nW2RhdGEtb3JpZW50YXRpb249J3ZlcnRpY2FsJ10gLnAtVGFiQmFyLXRhYixcbi8qIDwvREVQUkVDQVRFRD4gKi9cbi5sbS1UYWJCYXIubG0tbW9kLWRyYWdnaW5nW2RhdGEtb3JpZW50YXRpb249J3ZlcnRpY2FsJ10gLmxtLVRhYkJhci10YWIge1xuICB0b3A6IDA7XG4gIHRyYW5zaXRpb246IHRvcCAxNTBtcyBlYXNlO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLVRhYkJhci5wLW1vZC1kcmFnZ2luZyAucC1UYWJCYXItdGFiLnAtbW9kLWRyYWdnaW5nLFxuLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLVRhYkJhci5sbS1tb2QtZHJhZ2dpbmcgLmxtLVRhYkJhci10YWIubG0tbW9kLWRyYWdnaW5nIHtcbiAgdHJhbnNpdGlvbjogbm9uZTtcbn1cblxuLmxtLVRhYkJhci10YWJMYWJlbCAubG0tVGFiQmFyLXRhYklucHV0IHtcbiAgdXNlci1zZWxlY3Q6IGFsbDtcbiAgd2lkdGg6IDEwMCU7XG4gIGJveC1zaXppbmc6IGJvcmRlci1ib3g7XG4gIGJhY2tncm91bmQ6IGluaGVyaXQ7XG59XG4iXSwic291cmNlUm9vdCI6IiJ9 */</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/* <DEPRECATED> */
.p-TabPanel-tabBar, /* </DEPRECATED> */
.lm-TabPanel-tabBar {
  z-index: 1;
}

/* <DEPRECATED> */
.p-TabPanel-stackedPanel, /* </DEPRECATED> */
.lm-TabPanel-stackedPanel {
  z-index: 0;
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL25vZGVfbW9kdWxlcy9AbHVtaW5vL3dpZGdldHMvc3R5bGUvdGFicGFuZWwuY3NzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OzhFQU84RTs7QUFFOUUsaUJBQWlCO0FBQ2pCOztFQUVFLFVBQVU7QUFDWjs7QUFFQSxpQkFBaUI7QUFDakI7O0VBRUUsVUFBVTtBQUNaIiwic291cmNlc0NvbnRlbnQiOlsiLyotLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxufCBDb3B5cmlnaHQgKGMpIEp1cHl0ZXIgRGV2ZWxvcG1lbnQgVGVhbS5cbnwgQ29weXJpZ2h0IChjKSAyMDE0LTIwMTcsIFBob3NwaG9ySlMgQ29udHJpYnV0b3JzXG58XG58IERpc3RyaWJ1dGVkIHVuZGVyIHRoZSB0ZXJtcyBvZiB0aGUgQlNEIDMtQ2xhdXNlIExpY2Vuc2UuXG58XG58IFRoZSBmdWxsIGxpY2Vuc2UgaXMgaW4gdGhlIGZpbGUgTElDRU5TRSwgZGlzdHJpYnV0ZWQgd2l0aCB0aGlzIHNvZnR3YXJlLlxufC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0qL1xuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5wLVRhYlBhbmVsLXRhYkJhciwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmxtLVRhYlBhbmVsLXRhYkJhciB7XG4gIHotaW5kZXg6IDE7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLnAtVGFiUGFuZWwtc3RhY2tlZFBhbmVsLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4ubG0tVGFiUGFuZWwtc3RhY2tlZFBhbmVsIHtcbiAgei1pbmRleDogMDtcbn1cbiJdLCJzb3VyY2VSb290IjoiIn0= */</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL25vZGVfbW9kdWxlcy9AbHVtaW5vL3dpZGdldHMvc3R5bGUvaW5kZXguY3NzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OzhFQU84RSIsInNvdXJjZXNDb250ZW50IjpbIi8qLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbnwgQ29weXJpZ2h0IChjKSBKdXB5dGVyIERldmVsb3BtZW50IFRlYW0uXG58IENvcHlyaWdodCAoYykgMjAxNC0yMDE3LCBQaG9zcGhvckpTIENvbnRyaWJ1dG9yc1xufFxufCBEaXN0cmlidXRlZCB1bmRlciB0aGUgdGVybXMgb2YgdGhlIEJTRCAzLUNsYXVzZSBMaWNlbnNlLlxufFxufCBUaGUgZnVsbCBsaWNlbnNlIGlzIGluIHRoZSBmaWxlIExJQ0VOU0UsIGRpc3RyaWJ1dGVkIHdpdGggdGhpcyBzb2Z0d2FyZS5cbnwtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tKi9cbkBpbXBvcnQgJy4vd2lkZ2V0LmNzcyc7XG5AaW1wb3J0ICcuL2FjY29yZGlvbnBhbmVsLmNzcyc7XG5AaW1wb3J0ICcuL2NvbW1hbmRwYWxldHRlLmNzcyc7XG5AaW1wb3J0ICcuL2RvY2twYW5lbC5jc3MnO1xuQGltcG9ydCAnLi9tZW51LmNzcyc7XG5AaW1wb3J0ICcuL21lbnViYXIuY3NzJztcbkBpbXBvcnQgJy4vc2Nyb2xsYmFyLmNzcyc7XG5AaW1wb3J0ICcuL3NwbGl0cGFuZWwuY3NzJztcbkBpbXBvcnQgJy4vdGFiYmFyLmNzcyc7XG5AaW1wb3J0ICcuL3RhYnBhbmVsLmNzcyc7XG4iXSwic291cmNlUm9vdCI6IiJ9 */</style><style>/* Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jupyter-widgets-disconnected::before {
  content: '\f127'; /* chain-broken */
  display: inline-block;
  font: normal normal 900 14px/1 'Font Awesome 5 Free', 'FontAwesome';
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #d9534f;
  padding: 3px;
  align-self: flex-start;
}

.jupyter-widgets-error-widget {
  display: flex;
  flex-direction: column;
  justify-content: center;
  height: 100%;
  border: solid 1px red;
  margin: 0 auto;
}

.jupyter-widgets-error-widget.icon-error {
  min-width: var(--jp-widgets-inline-width-short);
}
.jupyter-widgets-error-widget.text-error {
  min-width: calc(2 * var(--jp-widgets-inline-width));
  min-height: calc(3 * var(--jp-widgets-inline-height));
}

.jupyter-widgets-error-widget p {
  text-align: center;
}

.jupyter-widgets-error-widget.text-error pre::first-line {
  font-weight: bold;
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL3BhY2thZ2VzL2Jhc2UvY3NzL2luZGV4LmNzcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7RUFFRTs7QUFFRjtFQUNFLGdCQUFnQixFQUFFLGlCQUFpQjtFQUNuQyxxQkFBcUI7RUFDckIsbUVBQW1FO0VBQ25FLG9CQUFvQjtFQUNwQixtQ0FBbUM7RUFDbkMsa0NBQWtDO0VBQ2xDLGNBQWM7RUFDZCxZQUFZO0VBQ1osc0JBQXNCO0FBQ3hCOztBQUVBO0VBQ0UsYUFBYTtFQUNiLHNCQUFzQjtFQUN0Qix1QkFBdUI7RUFDdkIsWUFBWTtFQUNaLHFCQUFxQjtFQUNyQixjQUFjO0FBQ2hCOztBQUVBO0VBQ0UsK0NBQStDO0FBQ2pEO0FBQ0E7RUFDRSxtREFBbUQ7RUFDbkQscURBQXFEO0FBQ3ZEOztBQUVBO0VBQ0Usa0JBQWtCO0FBQ3BCOztBQUVBO0VBQ0UsaUJBQWlCO0FBQ25CIiwic291cmNlc0NvbnRlbnQiOlsiLyogQ29weXJpZ2h0IChjKSBKdXB5dGVyIERldmVsb3BtZW50IFRlYW0uXG4gKiBEaXN0cmlidXRlZCB1bmRlciB0aGUgdGVybXMgb2YgdGhlIE1vZGlmaWVkIEJTRCBMaWNlbnNlLlxuICovXG5cbi5qdXB5dGVyLXdpZGdldHMtZGlzY29ubmVjdGVkOjpiZWZvcmUge1xuICBjb250ZW50OiAnXFxmMTI3JzsgLyogY2hhaW4tYnJva2VuICovXG4gIGRpc3BsYXk6IGlubGluZS1ibG9jaztcbiAgZm9udDogbm9ybWFsIG5vcm1hbCA5MDAgMTRweC8xICdGb250IEF3ZXNvbWUgNSBGcmVlJywgJ0ZvbnRBd2Vzb21lJztcbiAgdGV4dC1yZW5kZXJpbmc6IGF1dG87XG4gIC13ZWJraXQtZm9udC1zbW9vdGhpbmc6IGFudGlhbGlhc2VkO1xuICAtbW96LW9zeC1mb250LXNtb290aGluZzogZ3JheXNjYWxlO1xuICBjb2xvcjogI2Q5NTM0ZjtcbiAgcGFkZGluZzogM3B4O1xuICBhbGlnbi1zZWxmOiBmbGV4LXN0YXJ0O1xufVxuXG4uanVweXRlci13aWRnZXRzLWVycm9yLXdpZGdldCB7XG4gIGRpc3BsYXk6IGZsZXg7XG4gIGZsZXgtZGlyZWN0aW9uOiBjb2x1bW47XG4gIGp1c3RpZnktY29udGVudDogY2VudGVyO1xuICBoZWlnaHQ6IDEwMCU7XG4gIGJvcmRlcjogc29saWQgMXB4IHJlZDtcbiAgbWFyZ2luOiAwIGF1dG87XG59XG5cbi5qdXB5dGVyLXdpZGdldHMtZXJyb3Itd2lkZ2V0Lmljb24tZXJyb3Ige1xuICBtaW4td2lkdGg6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLXdpZHRoLXNob3J0KTtcbn1cbi5qdXB5dGVyLXdpZGdldHMtZXJyb3Itd2lkZ2V0LnRleHQtZXJyb3Ige1xuICBtaW4td2lkdGg6IGNhbGMoMiAqIHZhcigtLWpwLXdpZGdldHMtaW5saW5lLXdpZHRoKSk7XG4gIG1pbi1oZWlnaHQ6IGNhbGMoMyAqIHZhcigtLWpwLXdpZGdldHMtaW5saW5lLWhlaWdodCkpO1xufVxuXG4uanVweXRlci13aWRnZXRzLWVycm9yLXdpZGdldCBwIHtcbiAgdGV4dC1hbGlnbjogY2VudGVyO1xufVxuXG4uanVweXRlci13aWRnZXRzLWVycm9yLXdpZGdldC50ZXh0LWVycm9yIHByZTo6Zmlyc3QtbGluZSB7XG4gIGZvbnQtd2VpZ2h0OiBib2xkO1xufVxuIl0sInNvdXJjZVJvb3QiOiIifQ== */</style><style>/**
 * The material design colors are adapted from google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 * https://github.com/danlevan/google-material-color/blob/f67ca5f4028b2f1b34862f64b0ca67323f91b088/dist/palette.var.css
 *
 * The license for the material design color CSS variables is as follows (see
 * https://github.com/danlevan/google-material-color/blob/f67ca5f4028b2f1b34862f64b0ca67323f91b088/LICENSE)
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2014 Dan Le Van
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;

  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;

  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #aa00ff;

  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;

  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;

  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;

  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;

  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;

  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;

  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;

  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;

  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;

  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ffff00;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;

  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;

  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;

  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;

  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;

  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eeeeee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;

  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL3BhY2thZ2VzL2NvbnRyb2xzL2Nzcy9tYXRlcmlhbGNvbG9ycy5jc3MiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0VBNkJFO0FBQ0Y7RUFDRSxvQkFBb0I7RUFDcEIscUJBQXFCO0VBQ3JCLHFCQUFxQjtFQUNyQixxQkFBcUI7RUFDckIscUJBQXFCO0VBQ3JCLHFCQUFxQjtFQUNyQixxQkFBcUI7RUFDckIscUJBQXFCO0VBQ3JCLHFCQUFxQjtFQUNyQixxQkFBcUI7RUFDckIsc0JBQXNCO0VBQ3RCLHNCQUFzQjtFQUN0QixzQkFBc0I7RUFDdEIsc0JBQXNCOztFQUV0QixxQkFBcUI7RUFDckIsc0JBQXNCO0VBQ3RCLHNCQUFzQjtFQUN0QixzQkFBc0I7RUFDdEIsc0JBQXNCO0VBQ3RCLHNCQUFzQjtFQUN0QixzQkFBc0I7RUFDdEIsc0JBQXNCO0VBQ3RCLHNCQUFzQjtFQUN0QixzQkFBc0I7RUFDdEIsdUJBQXVCO0VBQ3ZCLHVCQUF1QjtFQUN2Qix1QkFBdUI7RUFDdkIsdUJBQXVCOztFQUV2Qix1QkFBdUI7RUFDdkIsd0JBQXdCO0VBQ3hCLHdCQUF3QjtFQUN4Qix3QkFBd0I7RUFDeEIsd0JBQXdCO0VBQ3hCLHdCQUF3QjtFQUN4Qix3QkFBd0I7RUFDeEIsd0JBQXdCO0VBQ3hCLHdCQUF3QjtFQUN4Qix3QkFBd0I7RUFDeEIseUJBQXlCO0VBQ3pCLHlCQUF5QjtFQUN6Qix5QkFBeUI7RUFDekIseUJBQXlCOztFQUV6Qiw0QkFBNEI7RUFDNUIsNkJBQTZCO0VBQzdCLDZCQUE2QjtFQUM3Qiw2QkFBNkI7RUFDN0IsNkJBQTZCO0VBQzdCLDZCQUE2QjtFQUM3Qiw2QkFBNkI7RUFDN0IsNkJBQTZCO0VBQzdCLDZCQUE2QjtFQUM3Qiw2QkFBNkI7RUFDN0IsOEJBQThCO0VBQzlCLDhCQUE4QjtFQUM5Qiw4QkFBOEI7RUFDOUIsOEJBQThCOztFQUU5Qix1QkFBdUI7RUFD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RUFDeEIsd0JBQXdCOztFQUV4Qix1QkFBdUI7RUFDdkIsd0JBQXdCO0VBQ3hCLHdCQUF3QjtFQUN4Qix3QkFBd0I7RUFDeEIsd0JBQXdCO0VBQ3hCLHdCQUF3QjtFQUN4Qix3QkFBd0I7RUFDeEIsd0JBQXdCO0VBQ3hCLHdCQUF3QjtFQUN4Qix3QkFBd0I7RUFDeEIseUJBQXlCO0VBQ3pCLHlCQUF5QjtFQUN6Qix5QkFBeUI7RUFDekIseUJBQXlCOztFQUV6Qiw0QkFBNEI7RUFDNUIsNkJBQTZCO0VBQzdCLDZCQUE2QjtFQUM3Qiw2QkFBNkI7RUFDN0IsNkJBQTZCO0VBQzdCLDZCQUE2QjtFQUM3Qiw2QkFBNkI7RUFDN0IsNkJBQTZCO0VBQzdCLDZCQUE2QjtFQUM3Qiw2QkFBNkI7RUFDN0IsOEJBQThCO0VBQzlCLDhCQUE4QjtFQUM5Qiw4QkFBOEI7RUFDOUIsOEJBQThCOztFQUU5QixzQkFBc0I7RUFDdEIsdUJBQXVCO0VBQ3ZCLHVCQUF1QjtFQUN2Qix1QkFBdUI7RUFDdkIsdUJBQXVCO0VBQ3ZCLHVCQUF1QjtFQUN2Qix1QkFBdUI7RUFDdkIsdUJBQXVCO0VBQ3ZCLHVCQUF1QjtFQUN2Qix1QkFBdUI7O0VBRXZCLHFCQUFxQjtFQUNyQixzQkFBc0I7RUFDdEIsc0JBQXNCO0VBQ3RCLHNCQUFzQjtFQUN0QixzQkFBc0I7RUFDdEIsc0JBQXNCO0VBQ3RCLHNCQUFzQjtFQUN0QixzQkFBc0I7RUFDdEIsc0JBQXNCO0VBQ3RCLHNCQUFzQjs7RUFFdEIsMEJBQTBCO0VBQzFCLDJCQUEyQjtFQUMzQiwyQkFBMkI7RUFDM0IsMkJBQTJCO0VBQzNCLDJCQUEyQjtFQUMzQiwyQkFBMkI7RUFDM0IsMkJBQTJCO0VBQzNCLDJCQUEyQjtFQUMzQiwyQkFBMkI7RUFDM0IsMkJBQTJCO0FBQzdCIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBUaGUgbWF0ZXJpYWwgZGVzaWduIGNvbG9ycyBhcmUgYWRhcHRlZCBmcm9tIGdvb2dsZS1tYXRlcmlhbC1jb2xvciB2MS4yLjZcbiAqIGh0dHBzOi8vZ2l0aHViLmNvbS9kYW5sZXZhbi9nb29nbGUtbWF0ZXJpYWwtY29sb3JcbiAqIGh0dHBzOi8vZ2l0aHViLmNvbS9kYW5sZXZhbi9nb29nbGUtbWF0ZXJpYWwtY29sb3IvYmxvYi9mNjdjYTVmNDAyOGIyZjFiMzQ4NjJmNjRiMGNhNjczMjNmOTFiMDg4L2Rpc3QvcGFsZXR0ZS52YXIuY3NzXG4gKlxuICogVGhlIGxpY2Vuc2UgZm9yIHRoZSBtYXRlcmlhbCBkZXNpZ24gY29sb3IgQ1NTIHZhcmlhYmxlcyBpcyBhcyBmb2xsb3dzIChzZWVcbiAqIGh0dHBzOi8vZ2l0aHViLmNvbS9kYW5sZXZhbi9nb29nbGUtbWF0ZXJpYWwtY29sb3IvYmxvYi9mNjdjYTVmNDAyOGIyZjFiMzQ4NjJmNjRiMGNhNjczMjNmOTFiMDg4L0xJQ0VOU0UpXG4gKlxuICogVGhlIE1JVCBMaWNlbnNlIChNSVQpXG4gKlxuICogQ29weXJpZ2h0IChjKSAyMDE0IERhbiBMZSBWYW5cbiAqXG4gKiBQZXJtaXNzaW9uIGlzIGhlcmVieSBncmFudGVkLCBmcmVlIG9mIGNoYXJnZSwgdG8gYW55IHBlcnNvbiBvYnRhaW5pbmcgYSBjb3B5XG4gKiBvZiB0aGlzIHNvZnR3YXJlIGFuZCBhc3NvY2lhdGVkIGRvY3VtZW50YXRpb24gZmlsZXMgKHRoZSBcIlNvZnR3YXJlXCIpLCB0byBkZWFsXG4gKiBpbiB0aGUgU29mdHdhcmUgd2l0aG91dCByZXN0cmljdGlvbiwgaW5jbHVkaW5nIHdpdGhvdXQgbGltaXRhdGlvbiB0aGUgcmlnaHRzXG4gKiB0byB1c2UsIGNvcHksIG1vZGlmeSwgbWVyZ2UsIHB1Ymxpc2gsIGRpc3RyaWJ1dGUsIHN1YmxpY2Vuc2UsIGFuZC9vciBzZWxsXG4gKiBjb3BpZXMgb2YgdGhlIFNvZnR3YXJlLCBhbmQgdG8gcGVybWl0IHBlcnNvbnMgdG8gd2hvbSB0aGUgU29mdHdhcmUgaXNcbiAqIGZ1cm5pc2hlZCB0byBkbyBzbywgc3ViamVjdCB0byB0aGUgZm9sbG93aW5nIGNvbmRpdGlvbnM6XG4gKlxuICogVGhlIGFib3ZlIGNvcHlyaWdodCBub3RpY2UgYW5kIHRoaXMgcGVybWlzc2lvbiBub3RpY2Ugc2hhbGwgYmUgaW5jbHVkZWQgaW5cbiAqIGFsbCBjb3BpZXMgb3Igc3Vic3RhbnRpYWwgcG9ydGlvbnMgb2YgdGhlIFNvZnR3YXJlLlxuICpcbiAqIFRIRSBTT0ZUV0FSRSBJUyBQUk9WSURFRCBcIkFTIElTXCIsIFdJVEhPVVQgV0FSUkFOVFkgT0YgQU5ZIEtJTkQsIEVYUFJFU1MgT1JcbiAqIElNUExJRUQsIElOQ0xVRElORyBCVVQgTk9UIExJTUlURUQgVE8gVEhFIFdBUlJBTlRJRVMgT0YgTUVSQ0hBTlRBQklMSVRZLFxuICogRklUTkVTUyBGT1IgQSBQQVJUSUNVTEFSIFBVUlBPU0UgQU5EIE5PTklORlJJTkdFTUVOVC4gSU4gTk8gRVZFTlQgU0hBTEwgVEhFXG4gKiBBVVRIT1JTIE9SIENPUFlSSUdIVCBIT0xERVJTIEJFIExJQUJMRSBGT1IgQU5ZIENMQUlNLCBEQU1BR0VTIE9SIE9USEVSXG4gKiBMSUFCSUxJVFksIFdIRVRIRVIgSU4gQU4gQUNUSU9OIE9GIENPTlRSQUNULCBUT1JUIE9SIE9USEVSV0lTRSwgQVJJU0lORyBGUk9NLFxuICogT1VUIE9GIE9SIElOIENPTk5FQ1RJT04gV0lUSCBUSEUgU09GVFdBUkUgT1IgVEhFIFVTRSBPUiBPVEhFUiBERUFMSU5HUyBJTiBUSEVcbiAqIFNPRlRXQVJFLlxuICovXG46cm9vdCB7XG4gIC0tbWQtcmVkLTUwOiAjZmZlYmVlO1xuICAtLW1kLXJlZC0xMDA6ICNmZmNkZDI7XG4gIC0tbWQtcmVkLTIwMDogI2VmOWE5YTtcbiAgLS1tZC1yZWQtMzAwOiAjZTU3MzczO1xuICAtLW1kLXJlZC00MDA6ICNlZjUzNTA7XG4gIC0tbWQtcmVkLTUwMDogI2Y0NDMzNjtcbiAgLS1tZC1yZWQtNjAwOiAjZTUzOTM1O1xuICAtLW1kLXJlZC03MDA6ICNkMzJmMmY7XG4gIC0tbWQtcmVkLTgwMDogI2M2MjgyODtcbiAgLS1tZC1yZWQtOTAwOiAjYjcxYzFjO1xuICAtLW1kLXJlZC1BMTAwOiAjZmY4YTgwO1xuICAtLW1kLXJlZC1BMjAwOiAjZmY1MjUyO1xuICAtLW1kLXJlZC1BNDAwOiAjZmYxNzQ0O1xuICAtLW1kLXJlZC1BNzAwOiAjZDUwMDAwO1xuXG4gIC0tbWQtcGluay01MDogI2ZjZTRlYztcbiAgLS1tZC1waW5rLTEwMDogI2Y4YmJkMDtcbiAgLS1tZC1waW5rLTIwMDogI2Y0OGZiMTtcbiAgLS1tZC1waW5rLTMwMDogI2YwNjI5MjtcbiAgLS1tZC1waW5rLTQwMDogI2VjNDA3YTtcbiAgLS1tZC1waW5rLTUwMDogI2U5MWU2MztcbiAgLS1tZC1waW5rLTYwMDogI2Q4MWI2MDtcbiAgLS1tZC1waW5rLTcwMDogI2MyMTg1YjtcbiAgLS1tZC1waW5rLTgwMDogI2FkMTQ1NztcbiAgLS1tZC1waW5rLTkwMDogIzg4MGU0ZjtcbiAgLS1tZC1waW5rLUExMDA6ICNmZjgwYWI7XG4gIC0tbWQtcGluay1BMjAwOiAjZmY0MDgxO1xuICAtLW1kLXBpbmstQTQwMDogI2Y1MDA1NztcbiAgLS1tZC1waW5rLUE3MDA6ICNjNTExNjI7XG5cbiAgLS1tZC1wdXJwbGUtNTA6ICNmM2U1ZjU7XG4gIC0tbWQtcHVycGxlLTEwMDogI2UxYmVlNztcbiAgLS1tZC1wdXJwbGUtMjAwOiAjY2U5M2Q4O1xuICAtLW1kLXB1cnBsZS0zMDA6ICNiYTY4Yzg7XG4gIC0tbWQtcHVycGxlLTQwMDogI2FiNDdiYztcbiAgLS1tZC1wdXJwbGUtNTAwOiAjOWMyN2IwO1xuICAtLW1kLXB1cnBsZS02MDA6ICM4ZTI0YWE7XG4gIC0tbWQtcHVycGxlLTcwMDogIzdiMWZhMjtcbiAgLS1tZC1wdXJwbGUtODAwOiAjNmExYjlhO1xuICAtLW1kLXB1cnBsZS05MDA6ICM0YTE0OGM7XG4gIC0tbWQtcHVycGxlLUExMDA6ICNlYTgwZmM7XG4gIC0tbWQtcHVycGxlLUEyMDA6ICNlMDQwZmI7XG4gIC0tbWQtcHVycGxlLUE0MDA6ICNkNTAwZjk7XG4gIC0tbWQtcHVycGxlLUE3MDA6ICNhYTAwZmY7XG5cbiAgLS1tZC1kZWVwLXB1cnBsZS01MDogI2VkZTdmNjtcbiAgLS1tZC1kZWVwLXB1cnBsZS0xMDA6ICNkMWM0ZTk7XG4gIC0tbWQtZGVlcC1wdXJwbGUtMjAwOiAjYjM5ZGRiO1xuICAtLW1kLWRlZXAtcHVycGxlLTMwMDogIzk1NzVjZDtcbiAgLS1tZC1kZWVwLXB1cnBsZS00MDA6ICM3ZTU3YzI7XG4gIC0tbWQtZGVlcC1wdXJwbGUtNTAwOiAjNjczYWI3O1xuICAtLW1kLWRlZXAtcHVycGxlLTYwMDogIzVlMzViMTtcbiAgLS1tZC1kZWVwLXB1cnBsZS03MDA6ICM1MTJkYTg7XG4gIC0tbWQtZGVlcC1wdXJwbGUtODAwOiAjNDUyN2EwO1xuICAtLW1kLWRlZXAtcHVycGxlLTkwMDogIzMxMWI5MjtcbiAgLS1tZC1kZWVwLXB1cnBsZS1BMTAwOiAjYjM4OGZmO1xuICAtLW1kLWRlZXAtcHVycGxlLUEyMDA6ICM3YzRkZmY7XG4gIC0tbWQtZGVlcC1wdXJwbGUtQTQwMDogIzY1MWZmZjtcbiAgLS1tZC1kZWVwLXB1cnBsZS1BNzAwOiAjNjIwMGVhO1xuXG4gIC0tbWQtaW5kaWdvLTUwOiAjZThlYWY2O1xuICAtLW1kLWluZGlnby0xMDA6ICNjNWNhZTk7XG4gIC0tbWQtaW5kaWdvLTIwMDogIzlmYThkYTtcbiAgLS1tZC1pbmRpZ28tMzAwOiAjNzk4NmNiO1xuICAtLW1kLWluZGlnby00MDA6ICM1YzZiYzA7XG4gIC0tbWQtaW5kaWdvLTUwMDogIzNmNTFiNTtcbiAgLS1tZC1pbmRpZ28tNjAwOiAjMzk0OWFiO1xuICAtLW1kLWluZGlnby03MDA6ICMzMDNmOWY7XG4gIC0tbWQtaW5kaWdvLTgwMDogIzI4MzU5MztcbiAgLS1tZC1pbmRpZ28tOTAwOiAjMWEyMzdlO1xuICAtLW1kLWluZGlnby1BMTAwOiAjOGM5ZWZmO1xuICAtLW1kLWluZGlnby1BMjAwOiAjNTM2ZGZlO1xuICAtLW1kLWluZGlnby1BNDAwOiAjM2Q1YWZlO1xuICAtLW1kLWluZGlnby1BNzAwOiAjMzA0ZmZlO1xuXG4gIC0tbWQtYmx1ZS01MDogI2UzZjJmZDtcbiAgLS1tZC1ibHVlLTEwMDogI2JiZGVmYjtcbiAgLS1tZC1ibHVlLTIwMDogIzkwY2FmOTtcbiAgLS1tZC1ibHVlLTMwMDogIzY0YjVmNjtcbiAgLS1tZC1ibHVlLTQwMDogIzQyYTVmNTtcbiAgLS1tZC1ibHVlLTUwMDogIzIxOTZmMztcbiAgLS1tZC1ibHVlLTYwMDogIzFlODhlNTtcbiAgLS1tZC1ibHVlLTcwMDogIzE5NzZkMjtcbiAgLS1tZC1ibHVlLTgwMDogIzE1NjVjMDtcbiAgLS1tZC1ibHVlLTkwMDogIzBkNDdhMTtcbiAgLS1tZC1ibHVlLUExMDA6ICM4MmIxZmY7XG4gIC0tbWQtYmx1ZS1BMjAwOiAjNDQ4YWZmO1xuICAtLW1kLWJsdWUtQTQwMDogIzI5NzlmZjtcbiAgLS1tZC1ibHVlLUE3MDA6ICMyOTYyZmY7XG5cbiAgLS1tZC1saWdodC1ibHVlLTUwOiAjZTFmNWZlO1xuICAtLW1kLWxpZ2h0LWJsdWUtMTAwOiAjYjNlNWZjO1xuICAtLW1kLWxpZ2h0LWJsdWUtMjAwOiAjODFkNGZhO1xuICAtLW1kLWxpZ2h0LWJsdWUtMzAwOiAjNGZjM2Y3O1xuICAtLW1kLWxpZ2h0LWJsdWUtNDAwOiAjMjliNmY2O1xuICAtLW1kLWxpZ2h0LWJsdWUtNTAwOiAjMDNhOWY0O1xuICAtLW1kLWxpZ2h0LWJsdWUtNjAwOiAjMDM5YmU1O1xuICAtLW1kLWxpZ2h0LWJsdWUtNzAwOiAjMDI4OGQxO1xuICAtLW1kLWxpZ2h0LWJsdWUtODAwOiAjMDI3N2JkO1xuICAtLW1kLWxpZ2h0LWJsdWUtOTAwOiAjMDE1NzliO1xuICAtLW1kLWxpZ2h0LWJsdWUtQTEwMDogIzgwZDhmZjtcbiAgLS1tZC1saWdodC1ibHVlLUEyMDA6ICM0MGM0ZmY7XG4gIC0tbWQtbGlnaHQtYmx1ZS1BNDAwOiAjMDBiMGZmO1xuICAtLW1kLWxpZ2h0LWJsdWUtQTcwMDogIzAwOTFlYTtcblxuICAtLW1kLWN5YW4tNTA6ICNlMGY3ZmE7XG4gIC0tbWQtY3lhbi0xMDA6ICNiMmViZjI7XG4gIC0tbWQtY3lhbi0yMDA6ICM4MGRlZWE7XG4gIC0tbWQtY3lhbi0zMDA6ICM0ZGQwZTE7XG4gIC0tbWQtY3lhbi00MDA6ICMyNmM2ZGE7XG4gIC0tbWQtY3lhbi01MDA6ICMwMGJjZDQ7XG4gIC0tbWQtY3lhbi02MDA6ICMwMGFjYzE7XG4gIC0tbWQtY3lhbi03MDA6ICMwMDk3YTc7XG4gIC0tbWQtY3lhbi04MDA6ICMwMDgzOGY7XG4gIC0tbWQtY3lhbi05MDA6ICMwMDYwNjQ7XG4gIC0tbWQtY3lhbi1BMTAwOiAjODRmZmZmO1xuICAtLW1kLWN5YW4tQTIwMDogIzE4ZmZmZjtcbiAgLS1tZC1jeWFuLUE0MDA6ICMwMGU1ZmY7XG4gIC0tbWQtY3lhbi1BNzAwOiAjMDBiOGQ0O1xuXG4gIC0tbWQtdGVhbC01MDogI2UwZjJmMTtcbiAgLS1tZC10ZWFsLTEwMDogI2IyZGZkYjtcbiAgLS1tZC10ZWFsLTIwMDogIzgwY2JjNDtcbiAgLS1tZC10ZWFsLTMwMDogIzRkYjZhYztcbiAgLS1tZC10ZWFsLTQwMDogIzI2YTY5YTtcbiAgLS1tZC10ZWFsLTUwMDogIzAwOTY4ODtcbiAgLS1tZC10ZWFsLTYwMDogIzAwODk3YjtcbiAgLS1tZC10ZWFsLTcwMDogIzAwNzk2YjtcbiAgLS1tZC10ZWFsLTgwMDogIzAwNjk1YztcbiAgLS1tZC10ZWFsLTkwMDogIzAwNGQ0MDtcbiAgLS1tZC10ZWFsLUExMDA6ICNhN2ZmZWI7XG4gIC0tbWQtdGVhbC1BMjAwOiAjNjRmZmRhO1xuICAtLW1kLXRlYWwtQTQwMDogIzFkZTliNjtcbiAgLS1tZC10ZWFsLUE3MDA6ICMwMGJmYTU7XG5cbiAgLS1tZC1ncmVlbi01MDogI2U4ZjVlOTtcbiAgLS1tZC1ncmVlbi0xMDA6ICNjOGU2Yzk7XG4gIC0tbWQtZ3JlZW4tMjAwOiAjYTVkNmE3O1xuICAtLW1kLWdyZWVuLTMwMDogIzgxYzc4NDtcbiAgLS1tZC1ncmVlbi00MDA6ICM2NmJiNmE7XG4gIC0tbWQtZ3JlZW4tNTAwOiAjNGNhZjUwO1xuICAtLW1kLWdyZWVuLTYwMDogIzQzYTA0NztcbiAgLS1tZC1ncmVlbi03MDA6ICMzODhlM2M7XG4gIC0tbWQtZ3JlZW4tODAwOiAjMmU3ZDMyO1xuICAtLW1kLWdyZWVuLTkwMDogIzFiNWUyMDtcbiAgLS1tZC1ncmVlbi1BMTAwOiAjYjlmNmNhO1xuICAtLW1kLWdyZWVuLUEyMDA6ICM2OWYwYWU7XG4gIC0tbWQtZ3JlZW4tQTQwMDogIzAwZTY3NjtcbiAgLS1tZC1ncmVlbi1BNzAwOiAjMDBjODUzO1xuXG4gIC0tbWQtbGlnaHQtZ3JlZW4tNTA6ICNmMWY4ZTk7XG4gIC0tbWQtbGlnaHQtZ3JlZW4tMTAwOiAjZGNlZGM4O1xuICAtLW1kLWxpZ2h0LWdyZWVuLTIwMDogI2M1ZTFhNTtcbiAgLS1tZC1saWdodC1ncmVlbi0zMDA6ICNhZWQ1ODE7XG4gIC0tbWQtbGlnaHQtZ3JlZW4tNDAwOiAjOWNjYzY1O1xuICAtLW1kLWxpZ2h0LWdyZWVuLTUwMDogIzhiYzM0YTtcbiAgLS1tZC1saWdodC1ncmVlbi02MDA6ICM3Y2IzNDI7XG4gIC0tbWQtbGlnaHQtZ3JlZW4tNzAwOiAjNjg5ZjM4O1xuICAtLW1kLWxpZ2h0LWdyZWVuLTgwMDogIzU1OGIyZjtcbiAgLS1tZC1saWdodC1ncmVlbi05MDA6ICMzMzY5MWU7XG4gIC0tbWQtbGlnaHQtZ3JlZW4tQTEwMDogI2NjZmY5MDtcbiAgLS1tZC1saWdodC1ncmVlbi1BMjAwOiAjYjJmZjU5O1xuICAtLW1kLWxpZ2h0LWdyZWVuLUE0MDA6ICM3NmZmMDM7XG4gIC0tbWQtbGlnaHQtZ3JlZW4tQTcwMDogIzY0ZGQxNztcblxuICAtLW1kLWxpbWUtNTA6ICNmOWZiZTc7XG4gIC0tbWQtbGltZS0xMDA6ICNmMGY0YzM7XG4gIC0tbWQtbGltZS0yMDA6ICNlNmVlOWM7XG4gIC0tbWQtbGltZS0zMDA6ICNkY2U3NzU7XG4gIC0tbWQtbGltZS00MDA6ICNkNGUxNTc7XG4gIC0tbWQtbGltZS01MDA6ICNjZGRjMzk7XG4gIC0tbWQtbGltZS02MDA6ICNjMGNhMzM7XG4gIC0tbWQtbGltZS03MDA6ICNhZmI0MmI7XG4gIC0tbWQtbGltZS04MDA6ICM5ZTlkMjQ7XG4gIC0tbWQtbGltZS05MDA6ICM4Mjc3MTc7XG4gIC0tbWQtbGltZS1BMTAwOiAjZjRmZjgxO1xuICAtLW1kLWxpbWUtQTIwMDogI2VlZmY0MTtcbiAgLS1tZC1saW1lLUE0MDA6ICNjNmZmMDA7XG4gIC0tbWQtbGltZS1BNzAwOiAjYWVlYTAwO1xuXG4gIC0tbWQteWVsbG93LTUwOiAjZmZmZGU3O1xuICAtLW1kLXllbGxvdy0xMDA6ICNmZmY5YzQ7XG4gIC0tbWQteWVsbG93LTIwMDogI2ZmZjU5ZDtcbiAgLS1tZC15ZWxsb3ctMzAwOiAjZmZmMTc2O1xuICAtLW1kLXllbGxvdy00MDA6ICNmZmVlNTg7XG4gIC0tbWQteWVsbG93LTUwMDogI2ZmZWIzYjtcbiAgLS1tZC15ZWxsb3ctNjAwOiAjZmRkODM1O1xuICAtLW1kLXllbGxvdy03MDA6ICNmYmMwMmQ7XG4gIC0tbWQteWVsbG93LTgwMDogI2Y5YTgyNTtcbiAgLS1tZC15ZWxsb3ctOTAwOiAjZjU3ZjE3O1xuICAtLW1kLXllbGxvdy1BMTAwOiAjZmZmZjhkO1xuICAtLW1kLXllbGxvdy1BMjAwOiAjZmZmZjAwO1xuICAtLW1kLXllbGxvdy1BNDAwOiAjZmZlYTAwO1xuICAtLW1kLXllbGxvdy1BNzAwOiAjZmZkNjAwO1xuXG4gIC0tbWQtYW1iZXItNTA6ICNmZmY4ZTE7XG4gIC0tbWQtYW1iZXItMTAwOiAjZmZlY2IzO1xuICAtLW1kLWFtYmVyLTIwMDogI2ZmZTA4MjtcbiAgLS1tZC1hbWJlci0zMDA6ICNmZmQ1NGY7XG4gIC0tbWQtYW1iZXItNDAwOiAjZmZjYTI4O1xuICAtLW1kLWFtYmVyLTUwMDogI2ZmYzEwNztcbiAgLS1tZC1hbWJlci02MDA6ICNmZmIzMDA7XG4gIC0tbWQtYW1iZXItNzAwOiAjZmZhMDAwO1xuICAtLW1kLWFtYmVyLTgwMDogI2ZmOGYwMDtcbiAgLS1tZC1hbWJlci05MDA6ICNmZjZmMDA7XG4gIC0tbWQtYW1iZXItQTEwMDogI2ZmZTU3ZjtcbiAgLS1tZC1hbWJlci1BMjAwOiAjZmZkNzQwO1xuICAtLW1kLWFtYmVyLUE0MDA6ICNmZmM0MDA7XG4gIC0tbWQtYW1iZXItQTcwMDogI2ZmYWIwMDtcblxuICAtLW1kLW9yYW5nZS01MDogI2ZmZjNlMDtcbiAgLS1tZC1vcmFuZ2UtMTAwOiAjZmZlMGIyO1xuICAtLW1kLW9yYW5nZS0yMDA6ICNmZmNjODA7XG4gIC0tbWQtb3JhbmdlLTMwMDogI2ZmYjc0ZDtcbiAgLS1tZC1vcmFuZ2UtNDAwOiAjZmZhNzI2O1xuICAtLW1kLW9yYW5nZS01MDA6ICNmZjk4MDA7XG4gIC0tbWQtb3JhbmdlLTYwMDogI2ZiOGMwMDtcbiAgLS1tZC1vcmFuZ2UtNzAwOiAjZjU3YzAwO1xuICAtLW1kLW9yYW5nZS04MDA6ICNlZjZjMDA7XG4gIC0tbWQtb3JhbmdlLTkwMDogI2U2NTEwMDtcbiAgLS1tZC1vcmFuZ2UtQTEwMDogI2ZmZDE4MDtcbiAgLS1tZC1vcmFuZ2UtQTIwMDogI2ZmYWI0MDtcbiAgLS1tZC1vcmFuZ2UtQTQwMDogI2ZmOTEwMDtcbiAgLS1tZC1vcmFuZ2UtQTcwMDogI2ZmNmQwMDtcblxuICAtLW1kLWRlZXAtb3JhbmdlLTUwOiAjZmJlOWU3O1xuICAtLW1kLWRlZXAtb3JhbmdlLTEwMDogI2ZmY2NiYztcbiAgLS1tZC1kZWVwLW9yYW5nZS0yMDA6ICNmZmFiOTE7XG4gIC0tbWQtZGVlcC1vcmFuZ2UtMzAwOiAjZmY4YTY1O1xuICAtLW1kLWRlZXAtb3JhbmdlLTQwMDogI2ZmNzA0MztcbiAgLS1tZC1kZWVwLW9yYW5nZS01MDA6ICNmZjU3MjI7XG4gIC0tbWQtZGVlcC1vcmFuZ2UtNjAwOiAjZjQ1MTFlO1xuICAtLW1kLWRlZXAtb3JhbmdlLTcwMDogI2U2NGExOTtcbiAgLS1tZC1kZWVwLW9yYW5nZS04MDA6ICNkODQzMTU7XG4gIC0tbWQtZGVlcC1vcmFuZ2UtOTAwOiAjYmYzNjBjO1xuICAtLW1kLWRlZXAtb3JhbmdlLUExMDA6ICNmZjllODA7XG4gIC0tbWQtZGVlcC1vcmFuZ2UtQTIwMDogI2ZmNmU0MDtcbiAgLS1tZC1kZWVwLW9yYW5nZS1BNDAwOiAjZmYzZDAwO1xuICAtLW1kLWRlZXAtb3JhbmdlLUE3MDA6ICNkZDJjMDA7XG5cbiAgLS1tZC1icm93bi01MDogI2VmZWJlOTtcbiAgLS1tZC1icm93bi0xMDA6ICNkN2NjYzg7XG4gIC0tbWQtYnJvd24tMjAwOiAjYmNhYWE0O1xuICAtLW1kLWJyb3duLTMwMDogI2ExODg3ZjtcbiAgLS1tZC1icm93bi00MDA6ICM4ZDZlNjM7XG4gIC0tbWQtYnJvd24tNTAwOiAjNzk1NTQ4O1xuICAtLW1kLWJyb3duLTYwMDogIzZkNGM0MTtcbiAgLS1tZC1icm93bi03MDA6ICM1ZDQwMzc7XG4gIC0tbWQtYnJvd24tODAwOiAjNGUzNDJlO1xuICAtLW1kLWJyb3duLTkwMDogIzNlMjcyMztcblxuICAtLW1kLWdyZXktNTA6ICNmYWZhZmE7XG4gIC0tbWQtZ3JleS0xMDA6ICNmNWY1ZjU7XG4gIC0tbWQtZ3JleS0yMDA6ICNlZWVlZWU7XG4gIC0tbWQtZ3JleS0zMDA6ICNlMGUwZTA7XG4gIC0tbWQtZ3JleS00MDA6ICNiZGJkYmQ7XG4gIC0tbWQtZ3JleS01MDA6ICM5ZTllOWU7XG4gIC0tbWQtZ3JleS02MDA6ICM3NTc1NzU7XG4gIC0tbWQtZ3JleS03MDA6ICM2MTYxNjE7XG4gIC0tbWQtZ3JleS04MDA6ICM0MjQyNDI7XG4gIC0tbWQtZ3JleS05MDA6ICMyMTIxMjE7XG5cbiAgLS1tZC1ibHVlLWdyZXktNTA6ICNlY2VmZjE7XG4gIC0tbWQtYmx1ZS1ncmV5LTEwMDogI2NmZDhkYztcbiAgLS1tZC1ibHVlLWdyZXktMjAwOiAjYjBiZWM1O1xuICAtLW1kLWJsdWUtZ3JleS0zMDA6ICM5MGE0YWU7XG4gIC0tbWQtYmx1ZS1ncmV5LTQwMDogIzc4OTA5YztcbiAgLS1tZC1ibHVlLWdyZXktNTAwOiAjNjA3ZDhiO1xuICAtLW1kLWJsdWUtZ3JleS02MDA6ICM1NDZlN2E7XG4gIC0tbWQtYmx1ZS1ncmV5LTcwMDogIzQ1NWE2NDtcbiAgLS1tZC1ibHVlLWdyZXktODAwOiAjMzc0NzRmO1xuICAtLW1kLWJsdWUtZ3JleS05MDA6ICMyNjMyMzg7XG59XG4iXSwic291cmNlUm9vdCI6IiJ9 */</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
This file is copied from the JupyterLab project to define default styling for
when the widget styling is compiled down to eliminate CSS variables. We make one
change - we comment out the font import below.
*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/

/*
 * Optional monospace font for input/output prompt.
 */
/* Commented out in ipywidgets since we don't need it. */
/* @import url('https://fonts.googleapis.com/css?family=Roboto+Mono'); */

/*
 * Added for compatibility with output area
 */
:root {
  --jp-icon-search: none;
  --jp-ui-select-caret: none;
}

:root {
  /* Borders

  The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-700);
  --jp-border-color1: var(--md-grey-500);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-100);

  /* UI Fonts

  The UI font CSS variables are used for the typography all of the JupyterLab
  user interface elements that are not directly user generated content.
  */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: calc(
    var(--jp-ui-font-size1) / var(--jp-ui-font-scale-factor)
  );
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: calc(
    var(--jp-ui-font-size1) * var(--jp-ui-font-scale-factor)
  );
  --jp-ui-font-size3: calc(
    var(--jp-ui-font-size2) * var(--jp-ui-font-scale-factor)
  );
  --jp-ui-icon-font-size: 14px; /* Ensures px perfect FontAwesome icons */
  --jp-ui-font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;

  /* Use these font colors against the corresponding main layout colors.
     In a light theme, these go from dark to light.
  */

  --jp-ui-font-color0: rgba(0, 0, 0, 1);
  --jp-ui-font-color1: rgba(0, 0, 0, 0.8);
  --jp-ui-font-color2: rgba(0, 0, 0, 0.5);
  --jp-ui-font-color3: rgba(0, 0, 0, 0.3);

  /* Use these against the brand/accent/warn/error colors.
     These will typically go from light to darker, in both a dark and light theme
   */

  --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
  --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);

  /* Content Fonts

  Content font variables are used for typography of user generated content.
  */

  --jp-content-font-size: 13px;
  --jp-content-line-height: 1.5;
  --jp-content-font-color0: black;
  --jp-content-font-color1: black;
  --jp-content-font-color2: var(--md-grey-700);
  --jp-content-font-color3: var(--md-grey-500);

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: calc(
    var(--jp-ui-font-size1) / var(--jp-ui-font-scale-factor)
  );
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: calc(
    var(--jp-ui-font-size1) * var(--jp-ui-font-scale-factor)
  );
  --jp-ui-font-size3: calc(
    var(--jp-ui-font-size2) * var(--jp-ui-font-scale-factor)
  );

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.307;
  --jp-code-padding: 5px;
  --jp-code-font-family: monospace;

  /* Layout

  The following are the main layout colors use in JupyterLab. In a light
  theme these would go from light to dark.
  */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-700);
  --jp-brand-color1: var(--md-blue-500);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);

  --jp-accent-color0: var(--md-green-700);
  --jp-accent-color1: var(--md-green-500);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-700);
  --jp-warn-color1: var(--md-orange-500);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);

  --jp-error-color0: var(--md-red-700);
  --jp-error-color1: var(--md-red-500);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);

  --jp-success-color0: var(--md-green-700);
  --jp-success-color1: var(--md-green-500);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);

  --jp-info-color0: var(--md-cyan-700);
  --jp-info-color1: var(--md-cyan-500);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;
  --jp-cell-editor-background: #f7f7f7;
  --jp-cell-editor-border-color: #cfcfcf;
  --jp-cell-editor-background-edit: var(--jp-ui-layout-color1);
  --jp-cell-editor-border-color-edit: var(--jp-brand-color1);
  --jp-cell-prompt-width: 100px;
  --jp-cell-prompt-font-family: 'Roboto Mono', monospace;
  --jp-cell-prompt-letter-spacing: 0px;
  --jp-cell-prompt-opacity: 1;
  --jp-cell-prompt-opacity-not-active: 0.4;
  --jp-cell-prompt-font-color-not-active: var(--md-grey-700);
  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307fc1;
  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #bf5b3d;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-scroll-padding: 100px;

  /* Console specific styles */

  --jp-console-background: var(--md-grey-100);

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--md-grey-400);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color0);
  --jp-toolbar-box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.24);
  --jp-toolbar-header-margin: 4px 4px 0px 4px;
  --jp-toolbar-active-background: var(--md-grey-300);
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL3BhY2thZ2VzL2NvbnRyb2xzL2Nzcy9sYWJ2YXJpYWJsZXMuY3NzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7OEVBRzhFOztBQUU5RTs7OztDQUlDOztBQUlEOzs7Ozs7Ozs7Ozs7Ozs7Ozs7O0NBbUJDOztBQUVEOztFQUVFO0FBQ0Ysd0RBQXdEO0FBQ3hELHdFQUF3RTs7QUFFeEU7O0VBRUU7QUFDRjtFQUNFLHNCQUFzQjtFQUN0QiwwQkFBMEI7QUFDNUI7O0FBRUE7RUFDRTs7O0lBR0U7O0VBRUYsc0JBQXNCO0VBQ3RCLHNDQUFzQztFQUN0QyxzQ0FBc0M7RUFDdEMsc0NBQXNDO0VBQ3RDLHNDQUFzQzs7RUFFdEM7Ozs7R0FJQzs7RUFFRCw4QkFBOEI7RUFDOUI7O0dBRUM7RUFDRCx3QkFBd0IsRUFBRSxtQkFBbUI7RUFDN0M7O0dBRUM7RUFDRDs7R0FFQztFQUNELDRCQUE0QixFQUFFLHlDQUF5QztFQUN2RSxtRUFBbUU7O0VBRW5FOztHQUVDOztFQUVELHFDQUFxQztFQUNyQyx1Q0FBdUM7RUFDdkMsdUNBQXVDO0VBQ3ZDLHVDQUF1Qzs7RUFFdkM7O0lBRUU7O0VBRUYsbURBQW1EO0VBQ25ELG1EQUFtRDtFQUNuRCxxREFBcUQ7RUFDckQscURBQXFEOztFQUVyRDs7O0dBR0M7O0VBRUQsNEJBQTRCO0VBQzVCLDZCQUE2QjtFQUM3QiwrQkFBK0I7RUFDL0IsK0JBQStCO0VBQy9CLDRDQUE0QztFQUM1Qyw0Q0FBNEM7O0VBRTVDLDhCQUE4QjtFQUM5Qjs7R0FFQztFQUNELHdCQUF3QixFQUFFLG1CQUFtQjtFQUM3Qzs7R0FFQztFQUNEOztHQUVDOztFQUVELHlCQUF5QjtFQUN6Qiw0QkFBNEI7RUFDNUIsc0JBQXNCO0VBQ3RCLGdDQUFnQzs7RUFFaEM7Ozs7R0FJQzs7RUFFRCx5QkFBeUI7RUFDekIseUJBQXlCO0VBQ3pCLHNDQUFzQztFQUN0QyxzQ0FBc0M7O0VBRXRDLGlCQUFpQjs7RUFFakIscUNBQXFDO0VBQ3JDLHFDQUFxQztFQUNyQyxxQ0FBcUM7RUFDckMscUNBQXFDOztFQUVyQyx1Q0FBdUM7RUFDdkMsdUNBQXVDO0VBQ3ZDLHVDQUF1QztFQUN2Qyx1Q0FBdUM7O0VBRXZDLDhDQUE4Qzs7RUFFOUMsc0NBQXNDO0VBQ3RDLHNDQUFzQztFQUN0QyxzQ0FBc0M7RUFDdEMsc0NBQXNDOztFQUV0QyxvQ0FBb0M7RUFDcEMsb0NBQW9DO0VBQ3BDLG9DQUFvQztFQUNwQyxvQ0FBb0M7O0VBRXBDLHdDQUF3QztFQUN4Qyx3Q0FBd0M7RUFDeEMsd0NBQXdDO0VBQ3hDLHdDQUF3Qzs7RUFFeEMsb0NBQW9DO0VBQ3BDLG9DQUFvQztFQUNwQyxvQ0FBb0M7RUFDcEMsb0NBQW9DOztFQUVwQyx5QkFBeUI7O0VBRXpCLHNCQUFzQjtFQUN0QixvQ0FBb0M7RUFDcEMsc0NBQXNDO0VBQ3RDLDREQUE0RDtFQUM1RCwwREFBMEQ7RUFDMUQsNkJBQTZCO0VBQzdCLHNEQUFzRDtFQUN0RCxvQ0FBb0M7RUFDcEMsMkJBQTJCO0VBQzNCLHdDQUF3QztFQUN4QywwREFBMEQ7RUFDMUQ7MkVBQ3lFO0VBQ3pFLHNDQUFzQztFQUN0Qzt1RUFDcUU7RUFDckUsdUNBQXVDOztFQUV2Qyw2QkFBNkI7O0VBRTdCLDJCQUEyQjtFQUMzQixtQ0FBbUM7O0VBRW5DLDRCQUE0Qjs7RUFFNUIsMkNBQTJDOztFQUUzQyw0QkFBNEI7O0VBRTVCLDZDQUE2QztFQUM3Qyw4QkFBOEI7RUFDOUIsZ0RBQWdEO0VBQ2hELDREQUE0RDtFQUM1RCwyQ0FBMkM7RUFDM0Msa0RBQWtEO0FBQ3BEIiwic291cmNlc0NvbnRlbnQiOlsiLyotLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxufCBDb3B5cmlnaHQgKGMpIEp1cHl0ZXIgRGV2ZWxvcG1lbnQgVGVhbS5cbnwgRGlzdHJpYnV0ZWQgdW5kZXIgdGhlIHRlcm1zIG9mIHRoZSBNb2RpZmllZCBCU0QgTGljZW5zZS5cbnwtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tKi9cblxuLypcblRoaXMgZmlsZSBpcyBjb3BpZWQgZnJvbSB0aGUgSnVweXRlckxhYiBwcm9qZWN0IHRvIGRlZmluZSBkZWZhdWx0IHN0eWxpbmcgZm9yXG53aGVuIHRoZSB3aWRnZXQgc3R5bGluZyBpcyBjb21waWxlZCBkb3duIHRvIGVsaW1pbmF0ZSBDU1MgdmFyaWFibGVzLiBXZSBtYWtlIG9uZVxuY2hhbmdlIC0gd2UgY29tbWVudCBvdXQgdGhlIGZvbnQgaW1wb3J0IGJlbG93LlxuKi9cblxuQGltcG9ydCAnLi9tYXRlcmlhbGNvbG9ycy5jc3MnO1xuXG4vKlxuVGhlIGZvbGxvd2luZyBDU1MgdmFyaWFibGVzIGRlZmluZSB0aGUgbWFpbiwgcHVibGljIEFQSSBmb3Igc3R5bGluZyBKdXB5dGVyTGFiLlxuVGhlc2UgdmFyaWFibGVzIHNob3VsZCBiZSB1c2VkIGJ5IGFsbCBwbHVnaW5zIHdoZXJldmVyIHBvc3NpYmxlLiBJbiBvdGhlclxud29yZHMsIHBsdWdpbnMgc2hvdWxkIG5vdCBkZWZpbmUgY3VzdG9tIGNvbG9ycywgc2l6ZXMsIGV0YyB1bmxlc3MgYWJzb2x1dGVseVxubmVjZXNzYXJ5LiBUaGlzIGVuYWJsZXMgdXNlcnMgdG8gY2hhbmdlIHRoZSB2aXN1YWwgdGhlbWUgb2YgSnVweXRlckxhYlxuYnkgY2hhbmdpbmcgdGhlc2UgdmFyaWFibGVzLlxuXG5NYW55IHZhcmlhYmxlcyBhcHBlYXIgaW4gYW4gb3JkZXJlZCBzZXF1ZW5jZSAoMCwxLDIsMykuIFRoZXNlIHNlcXVlbmNlc1xuYXJlIGRlc2lnbmVkIHRvIHdvcmsgd2VsbCB0b2dldGhlciwgc28gZm9yIGV4YW1wbGUsIGAtLWpwLWJvcmRlci1jb2xvcjFgIHNob3VsZFxuYmUgdXNlZCB3aXRoIGAtLWpwLWxheW91dC1jb2xvcjFgLiBUaGUgbnVtYmVycyBoYXZlIHRoZSBmb2xsb3dpbmcgbWVhbmluZ3M6XG5cbiogMDogc3VwZXItcHJpbWFyeSwgcmVzZXJ2ZWQgZm9yIHNwZWNpYWwgZW1waGFzaXNcbiogMTogcHJpbWFyeSwgbW9zdCBpbXBvcnRhbnQgdW5kZXIgbm9ybWFsIHNpdHVhdGlvbnNcbiogMjogc2Vjb25kYXJ5LCBuZXh0IG1vc3QgaW1wb3J0YW50IHVuZGVyIG5vcm1hbCBzaXR1YXRpb25zXG4qIDM6IHRlcnRpYXJ5LCBuZXh0IG1vc3QgaW1wb3J0YW50IHVuZGVyIG5vcm1hbCBzaXR1YXRpb25zXG5cblRocm91Z2hvdXQgSnVweXRlckxhYiwgd2UgYXJlIG1vc3RseSBmb2xsb3dpbmcgcHJpbmNpcGxlcyBmcm9tIEdvb2dsZSdzXG5NYXRlcmlhbCBEZXNpZ24gd2hlbiBzZWxlY3RpbmcgY29sb3JzLiBXZSBhcmUgbm90LCBob3dldmVyLCBmb2xsb3dpbmdcbmFsbCBvZiBNRCBhcyBpdCBpcyBub3Qgb3B0aW1pemVkIGZvciBkZW5zZSwgaW5mb3JtYXRpb24gcmljaCBVSXMuXG4qL1xuXG4vKlxuICogT3B0aW9uYWwgbW9ub3NwYWNlIGZvbnQgZm9yIGlucHV0L291dHB1dCBwcm9tcHQuXG4gKi9cbi8qIENvbW1lbnRlZCBvdXQgaW4gaXB5d2lkZ2V0cyBzaW5jZSB3ZSBkb24ndCBuZWVkIGl0LiAqL1xuLyogQGltcG9ydCB1cmwoJ2h0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20vY3NzP2ZhbWlseT1Sb2JvdG8rTW9ubycpOyAqL1xuXG4vKlxuICogQWRkZWQgZm9yIGNvbXBhdGliaWxpdHkgd2l0aCBvdXRwdXQgYXJlYVxuICovXG46cm9vdCB7XG4gIC0tanAtaWNvbi1zZWFyY2g6IG5vbmU7XG4gIC0tanAtdWktc2VsZWN0LWNhcmV0OiBub25lO1xufVxuXG46cm9vdCB7XG4gIC8qIEJvcmRlcnNcblxuICBUaGUgZm9sbG93aW5nIHZhcmlhYmxlcywgc3BlY2lmeSB0aGUgdmlzdWFsIHN0eWxpbmcgb2YgYm9yZGVycyBpbiBKdXB5dGVyTGFiLlxuICAgKi9cblxuICAtLWpwLWJvcmRlci13aWR0aDogMXB4O1xuICAtLWpwLWJvcmRlci1jb2xvcjA6IHZhcigtLW1kLWdyZXktNzAwKTtcbiAgLS1qcC1ib3JkZXItY29sb3IxOiB2YXIoLS1tZC1ncmV5LTUwMCk7XG4gIC0tanAtYm9yZGVyLWNvbG9yMjogdmFyKC0tbWQtZ3JleS0zMDApO1xuICAtLWpwLWJvcmRlci1jb2xvcjM6IHZhcigtLW1kLWdyZXktMTAwKTtcblxuICAvKiBVSSBGb250c1xuXG4gIFRoZSBVSSBmb250IENTUyB2YXJpYWJsZXMgYXJlIHVzZWQgZm9yIHRoZSB0eXBvZ3JhcGh5IGFsbCBvZiB0aGUgSnVweXRlckxhYlxuICB1c2VyIGludGVyZmFjZSBlbGVtZW50cyB0aGF0IGFyZSBub3QgZGlyZWN0bHkgdXNlciBnZW5lcmF0ZWQgY29udGVudC5cbiAgKi9cblxuICAtLWpwLXVpLWZvbnQtc2NhbGUtZmFjdG9yOiAxLjI7XG4gIC0tanAtdWktZm9udC1zaXplMDogY2FsYyhcbiAgICB2YXIoLS1qcC11aS1mb250LXNpemUxKSAvIHZhcigtLWpwLXVpLWZvbnQtc2NhbGUtZmFjdG9yKVxuICApO1xuICAtLWpwLXVpLWZvbnQtc2l6ZTE6IDEzcHg7IC8qIEJhc2UgZm9udCBzaXplICovXG4gIC0tanAtdWktZm9udC1zaXplMjogY2FsYyhcbiAgICB2YXIoLS1qcC11aS1mb250LXNpemUxKSAqIHZhcigtLWpwLXVpLWZvbnQtc2NhbGUtZmFjdG9yKVxuICApO1xuICAtLWpwLXVpLWZvbnQtc2l6ZTM6IGNhbGMoXG4gICAgdmFyKC0tanAtdWktZm9udC1zaXplMikgKiB2YXIoLS1qcC11aS1mb250LXNjYWxlLWZhY3RvcilcbiAgKTtcbiAgLS1qcC11aS1pY29uLWZvbnQtc2l6ZTogMTRweDsgLyogRW5zdXJlcyBweCBwZXJmZWN0IEZvbnRBd2Vzb21lIGljb25zICovXG4gIC0tanAtdWktZm9udC1mYW1pbHk6ICdIZWx2ZXRpY2EgTmV1ZScsIEhlbHZldGljYSwgQXJpYWwsIHNhbnMtc2VyaWY7XG5cbiAgLyogVXNlIHRoZXNlIGZvbnQgY29sb3JzIGFnYWluc3QgdGhlIGNvcnJlc3BvbmRpbmcgbWFpbiBsYXlvdXQgY29sb3JzLlxuICAgICBJbiBhIGxpZ2h0IHRoZW1lLCB0aGVzZSBnbyBmcm9tIGRhcmsgdG8gbGlnaHQuXG4gICovXG5cbiAgLS1qcC11aS1mb250LWNvbG9yMDogcmdiYSgwLCAwLCAwLCAxKTtcbiAgLS1qcC11aS1mb250LWNvbG9yMTogcmdiYSgwLCAwLCAwLCAwLjgpO1xuICAtLWpwLXVpLWZvbnQtY29sb3IyOiByZ2JhKDAsIDAsIDAsIDAuNSk7XG4gIC0tanAtdWktZm9udC1jb2xvcjM6IHJnYmEoMCwgMCwgMCwgMC4zKTtcblxuICAvKiBVc2UgdGhlc2UgYWdhaW5zdCB0aGUgYnJhbmQvYWNjZW50L3dhcm4vZXJyb3IgY29sb3JzLlxuICAgICBUaGVzZSB3aWxsIHR5cGljYWxseSBnbyBmcm9tIGxpZ2h0IHRvIGRhcmtlciwgaW4gYm90aCBhIGRhcmsgYW5kIGxpZ2h0IHRoZW1lXG4gICAqL1xuXG4gIC0tanAtdWktaW52ZXJzZS1mb250LWNvbG9yMDogcmdiYSgyNTUsIDI1NSwgMjU1LCAxKTtcbiAgLS1qcC11aS1pbnZlcnNlLWZvbnQtY29sb3IxOiByZ2JhKDI1NSwgMjU1LCAyNTUsIDEpO1xuICAtLWpwLXVpLWludmVyc2UtZm9udC1jb2xvcjI6IHJnYmEoMjU1LCAyNTUsIDI1NSwgMC43KTtcbiAgLS1qcC11aS1pbnZlcnNlLWZvbnQtY29sb3IzOiByZ2JhKDI1NSwgMjU1LCAyNTUsIDAuNSk7XG5cbiAgLyogQ29udGVudCBGb250c1xuXG4gIENvbnRlbnQgZm9udCB2YXJpYWJsZXMgYXJlIHVzZWQgZm9yIHR5cG9ncmFwaHkgb2YgdXNlciBnZW5lcmF0ZWQgY29udGVudC5cbiAgKi9cblxuICAtLWpwLWNvbnRlbnQtZm9udC1zaXplOiAxM3B4O1xuICAtLWpwLWNvbnRlbnQtbGluZS1oZWlnaHQ6IDEuNTtcbiAgLS1qcC1jb250ZW50LWZvbnQtY29sb3IwOiBibGFjaztcbiAgLS1qcC1jb250ZW50LWZvbnQtY29sb3IxOiBibGFjaztcbiAgLS1qcC1jb250ZW50LWZvbnQtY29sb3IyOiB2YXIoLS1tZC1ncmV5LTcwMCk7XG4gIC0tanAtY29udGVudC1mb250LWNvbG9yMzogdmFyKC0tbWQtZ3JleS01MDApO1xuXG4gIC0tanAtdWktZm9udC1zY2FsZS1mYWN0b3I6IDEuMjtcbiAgLS1qcC11aS1mb250LXNpemUwOiBjYWxjKFxuICAgIHZhcigtLWpwLXVpLWZvbnQtc2l6ZTEpIC8gdmFyKC0tanAtdWktZm9udC1zY2FsZS1mYWN0b3IpXG4gICk7XG4gIC0tanAtdWktZm9udC1zaXplMTogMTNweDsgLyogQmFzZSBmb250IHNpemUgKi9cbiAgLS1qcC11aS1mb250LXNpemUyOiBjYWxjKFxuICAgIHZhcigtLWpwLXVpLWZvbnQtc2l6ZTEpICogdmFyKC0tanAtdWktZm9udC1zY2FsZS1mYWN0b3IpXG4gICk7XG4gIC0tanAtdWktZm9udC1zaXplMzogY2FsYyhcbiAgICB2YXIoLS1qcC11aS1mb250LXNpemUyKSAqIHZhcigtLWpwLXVpLWZvbnQtc2NhbGUtZmFjdG9yKVxuICApO1xuXG4gIC0tanAtY29kZS1mb250LXNpemU6IDEzcHg7XG4gIC0tanAtY29kZS1saW5lLWhlaWdodDogMS4zMDc7XG4gIC0tanAtY29kZS1wYWRkaW5nOiA1cHg7XG4gIC0tanAtY29kZS1mb250LWZhbWlseTogbW9ub3NwYWNlO1xuXG4gIC8qIExheW91dFxuXG4gIFRoZSBmb2xsb3dpbmcgYXJlIHRoZSBtYWluIGxheW91dCBjb2xvcnMgdXNlIGluIEp1cHl0ZXJMYWIuIEluIGEgbGlnaHRcbiAgdGhlbWUgdGhlc2Ugd291bGQgZ28gZnJvbSBsaWdodCB0byBkYXJrLlxuICAqL1xuXG4gIC0tanAtbGF5b3V0LWNvbG9yMDogd2hpdGU7XG4gIC0tanAtbGF5b3V0LWNvbG9yMTogd2hpdGU7XG4gIC0tanAtbGF5b3V0LWNvbG9yMjogdmFyKC0tbWQtZ3JleS0yMDApO1xuICAtLWpwLWxheW91dC1jb2xvcjM6IHZhcigtLW1kLWdyZXktNDAwKTtcblxuICAvKiBCcmFuZC9hY2NlbnQgKi9cblxuICAtLWpwLWJyYW5kLWNvbG9yMDogdmFyKC0tbWQtYmx1ZS03MDApO1xuICAtLWpwLWJyYW5kLWNvbG9yMTogdmFyKC0tbWQtYmx1ZS01MDApO1xuICAtLWpwLWJyYW5kLWNvbG9yMjogdmFyKC0tbWQtYmx1ZS0zMDApO1xuICAtLWpwLWJyYW5kLWNvbG9yMzogdmFyKC0tbWQtYmx1ZS0xMDApO1xuXG4gIC0tanAtYWNjZW50LWNvbG9yMDogdmFyKC0tbWQtZ3JlZW4tNzAwKTtcbiAgLS1qcC1hY2NlbnQtY29sb3IxOiB2YXIoLS1tZC1ncmVlbi01MDApO1xuICAtLWpwLWFjY2VudC1jb2xvcjI6IHZhcigtLW1kLWdyZWVuLTMwMCk7XG4gIC0tanAtYWNjZW50LWNvbG9yMzogdmFyKC0tbWQtZ3JlZW4tMTAwKTtcblxuICAvKiBTdGF0ZSBjb2xvcnMgKHdhcm4sIGVycm9yLCBzdWNjZXNzLCBpbmZvKSAqL1xuXG4gIC0tanAtd2Fybi1jb2xvcjA6IHZhcigtLW1kLW9yYW5nZS03MDApO1xuICAtLWpwLXdhcm4tY29sb3IxOiB2YXIoLS1tZC1vcmFuZ2UtNTAwKTtcbiAgLS1qcC13YXJuLWNvbG9yMjogdmFyKC0tbWQtb3JhbmdlLTMwMCk7XG4gIC0tanAtd2Fybi1jb2xvcjM6IHZhcigtLW1kLW9yYW5nZS0xMDApO1xuXG4gIC0tanAtZXJyb3ItY29sb3IwOiB2YXIoLS1tZC1yZWQtNzAwKTtcbiAgLS1qcC1lcnJvci1jb2xvcjE6IHZhcigtLW1kLXJlZC01MDApO1xuICAtLWpwLWVycm9yLWNvbG9yMjogdmFyKC0tbWQtcmVkLTMwMCk7XG4gIC0tanAtZXJyb3ItY29sb3IzOiB2YXIoLS1tZC1yZWQtMTAwKTtcblxuICAtLWpwLXN1Y2Nlc3MtY29sb3IwOiB2YXIoLS1tZC1ncmVlbi03MDApO1xuICAtLWpwLXN1Y2Nlc3MtY29sb3IxOiB2YXIoLS1tZC1ncmVlbi01MDApO1xuICAtLWpwLXN1Y2Nlc3MtY29sb3IyOiB2YXIoLS1tZC1ncmVlbi0zMDApO1xuICAtLWpwLXN1Y2Nlc3MtY29sb3IzOiB2YXIoLS1tZC1ncmVlbi0xMDApO1xuXG4gIC0tanAtaW5mby1jb2xvcjA6IHZhcigtLW1kLWN5YW4tNzAwKTtcbiAgLS1qcC1pbmZvLWNvbG9yMTogdmFyKC0tbWQtY3lhbi01MDApO1xuICAtLWpwLWluZm8tY29sb3IyOiB2YXIoLS1tZC1jeWFuLTMwMCk7XG4gIC0tanAtaW5mby1jb2xvcjM6IHZhcigtLW1kLWN5YW4tMTAwKTtcblxuICAvKiBDZWxsIHNwZWNpZmljIHN0eWxlcyAqL1xuXG4gIC0tanAtY2VsbC1wYWRkaW5nOiA1cHg7XG4gIC0tanAtY2VsbC1lZGl0b3ItYmFja2dyb3VuZDogI2Y3ZjdmNztcbiAgLS1qcC1jZWxsLWVkaXRvci1ib3JkZXItY29sb3I6ICNjZmNmY2Y7XG4gIC0tanAtY2VsbC1lZGl0b3ItYmFja2dyb3VuZC1lZGl0OiB2YXIoLS1qcC11aS1sYXlvdXQtY29sb3IxKTtcbiAgLS1qcC1jZWxsLWVkaXRvci1ib3JkZXItY29sb3ItZWRpdDogdmFyKC0tanAtYnJhbmQtY29sb3IxKTtcbiAgLS1qcC1jZWxsLXByb21wdC13aWR0aDogMTAwcHg7XG4gIC0tanAtY2VsbC1wcm9tcHQtZm9udC1mYW1pbHk6ICdSb2JvdG8gTW9ubycsIG1vbm9zcGFjZTtcbiAgLS1qcC1jZWxsLXByb21wdC1sZXR0ZXItc3BhY2luZzogMHB4O1xuICAtLWpwLWNlbGwtcHJvbXB0LW9wYWNpdHk6IDE7XG4gIC0tanAtY2VsbC1wcm9tcHQtb3BhY2l0eS1ub3QtYWN0aXZlOiAwLjQ7XG4gIC0tanAtY2VsbC1wcm9tcHQtZm9udC1jb2xvci1ub3QtYWN0aXZlOiB2YXIoLS1tZC1ncmV5LTcwMCk7XG4gIC8qIEEgY3VzdG9tIGJsZW5kIG9mIE1EIGdyZXkgYW5kIGJsdWUgNjAwXG4gICAqIFNlZSBodHRwczovL21leWVyd2ViLmNvbS9lcmljL3Rvb2xzL2NvbG9yLWJsZW5kLyM1NDZFN0E6MUU4OEU1OjU6aGV4ICovXG4gIC0tanAtY2VsbC1pbnByb21wdC1mb250LWNvbG9yOiAjMzA3ZmMxO1xuICAvKiBBIGN1c3RvbSBibGVuZCBvZiBNRCBncmV5IGFuZCBvcmFuZ2UgNjAwXG4gICAqIGh0dHBzOi8vbWV5ZXJ3ZWIuY29tL2VyaWMvdG9vbHMvY29sb3ItYmxlbmQvIzU0NkU3QTpGNDUxMUU6NTpoZXggKi9cbiAgLS1qcC1jZWxsLW91dHByb21wdC1mb250LWNvbG9yOiAjYmY1YjNkO1xuXG4gIC8qIE5vdGVib29rIHNwZWNpZmljIHN0eWxlcyAqL1xuXG4gIC0tanAtbm90ZWJvb2stcGFkZGluZzogMTBweDtcbiAgLS1qcC1ub3RlYm9vay1zY3JvbGwtcGFkZGluZzogMTAwcHg7XG5cbiAgLyogQ29uc29sZSBzcGVjaWZpYyBzdHlsZXMgKi9cblxuICAtLWpwLWNvbnNvbGUtYmFja2dyb3VuZDogdmFyKC0tbWQtZ3JleS0xMDApO1xuXG4gIC8qIFRvb2xiYXIgc3BlY2lmaWMgc3R5bGVzICovXG5cbiAgLS1qcC10b29sYmFyLWJvcmRlci1jb2xvcjogdmFyKC0tbWQtZ3JleS00MDApO1xuICAtLWpwLXRvb2xiYXItbWljcm8taGVpZ2h0OiA4cHg7XG4gIC0tanAtdG9vbGJhci1iYWNrZ3JvdW5kOiB2YXIoLS1qcC1sYXlvdXQtY29sb3IwKTtcbiAgLS1qcC10b29sYmFyLWJveC1zaGFkb3c6IDBweCAwcHggMnB4IDBweCByZ2JhKDAsIDAsIDAsIDAuMjQpO1xuICAtLWpwLXRvb2xiYXItaGVhZGVyLW1hcmdpbjogNHB4IDRweCAwcHggNHB4O1xuICAtLWpwLXRvb2xiYXItYWN0aXZlLWJhY2tncm91bmQ6IHZhcigtLW1kLWdyZXktMzAwKTtcbn1cbiJdLCJzb3VyY2VSb290IjoiIn0= */</style><style>/* This file has code derived from Lumino CSS files, as noted below. The license for this Lumino code is:

Copyright (c) 2019 Project Jupyter Contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Copyright (c) 2014-2017, PhosphorJS Contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 * The following section is derived from https://github.com/jupyterlab/lumino/blob/23b9d075ebc5b73ab148b6ebfc20af97f85714c4/packages/widgets/style/tabbar.css 
 * We've scoped the rules so that they are consistent with exactly our code.
 */

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar[data-orientation='horizontal'], /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar[data-orientation='horizontal'], /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar[data-orientation='vertical'], /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar[data-orientation='vertical'], /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar > .p-TabBar-content, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar > .p-TabBar-content, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar > .lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab
  > .p-TabBar[data-orientation='horizontal']
  > .p-TabBar-content,
/* </DEPRECATED> */
/* <DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
> .p-TabBar[data-orientation='horizontal']
> .p-TabBar-content,
/* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
  > .lm-TabBar[data-orientation='horizontal']
  > .lm-TabBar-content {
  flex-direction: row;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab
  > .p-TabBar[data-orientation='vertical']
  > .p-TabBar-content,
/* </DEPRECATED> */
/* <DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
> .p-TabBar[data-orientation='vertical']
> .p-TabBar-content,
/* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
  > .lm-TabBar[data-orientation='vertical']
  > .lm-TabBar-content {
  flex-direction: column;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar .p-TabBar-tab, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabIcon, /* </DEPRECATED> */
/* <DEPRECATED> */ .jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabCloseIcon, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar .p-TabBar-tabIcon, /* </DEPRECATED> */
/* <DEPRECATED> */ .jupyter-widgets.jupyter-widget-tab > .p-TabBar .p-TabBar-tabCloseIcon, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tabIcon,
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabLabel, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar .p-TabBar-tabLabel, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab.p-mod-hidden, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar .p-TabBar-tab.p-mod-hidden, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar.p-mod-dragging .p-TabBar-tab, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar.p-mod-dragging .p-TabBar-tab, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab
  > .p-TabBar.p-mod-dragging[data-orientation='horizontal']
  .p-TabBar-tab,
/* </DEPRECATED> */
/* <DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
  > .p-TabBar.p-mod-dragging[data-orientation='horizontal']
  .p-TabBar-tab,
/* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
  > .lm-TabBar.lm-mod-dragging[data-orientation='horizontal']
  .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab
  > .p-TabBar.p-mod-dragging[data-orientation='vertical']
  .p-TabBar-tab,
/* </DEPRECATED> */
/* <DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
> .p-TabBar.p-mod-dragging[data-orientation='vertical']
.p-TabBar-tab,
/* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
  > .lm-TabBar.lm-mod-dragging[data-orientation='vertical']
  .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab
  > .p-TabBar.p-mod-dragging
  .p-TabBar-tab.p-mod-dragging,
/* </DEPRECATED> */
/* <DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
> .p-TabBar.p-mod-dragging
.p-TabBar-tab.p-mod-dragging,
/* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
  > .lm-TabBar.lm-mod-dragging
  .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

/* End tabbar.css */

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL3BhY2thZ2VzL2NvbnRyb2xzL2Nzcy9sdW1pbm8uY3NzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0NBMERDOztBQUVEOzs7RUFHRTs7QUFFRixpQkFBaUI7QUFDakI7OztFQUdFLGFBQWE7RUFDYix5QkFBeUI7RUFDekIsc0JBQXNCO0VBQ3RCLHFCQUFxQjtFQUNyQixpQkFBaUI7QUFDbkI7O0FBRUEsaUJBQWlCO0FBQ2pCOzs7RUFHRSxtQkFBbUI7QUFDckI7O0FBRUEsaUJBQWlCO0FBQ2pCOzs7RUFHRSxzQkFBc0I7QUFDeEI7O0FBRUEsaUJBQWlCO0FBQ2pCOzs7RUFHRSxTQUFTO0VBQ1QsVUFBVTtFQUNWLGFBQWE7RUFDYixjQUFjO0VBQ2QscUJBQXFCO0FBQ3ZCOztBQUVBLGlCQUFpQjtBQUNqQjs7Ozs7Ozs7Ozs7O0VBWUUsbUJBQW1CO0FBQ3JCOztBQUVBLGlCQUFpQjtBQUNqQjs7Ozs7Ozs7Ozs7O0VBWUUsc0JBQXNCO0FBQ3hCOztBQUVBLGlCQUFpQjtBQUNqQjs7O0VBR0UsYUFBYTtFQUNiLG1CQUFtQjtFQUNuQixzQkFBc0I7RUFDdEIsZ0JBQWdCO0FBQ2xCOztBQUVBLGlCQUFpQjtBQUNqQjs7Ozs7O0VBTUUsY0FBYztBQUNoQjs7QUFFQSxpQkFBaUI7QUFDakI7OztFQUdFLGNBQWM7RUFDZCxnQkFBZ0I7RUFDaEIsbUJBQW1CO0FBQ3JCOztBQUVBLGlCQUFpQjtBQUNqQjs7O0VBR0Usd0JBQXdCO0FBQzFCOztBQUVBLGlCQUFpQjtBQUNqQjs7O0VBR0Usa0JBQWtCO0FBQ3BCOztBQUVBLGlCQUFpQjtBQUNqQjs7Ozs7Ozs7Ozs7O0VBWUUsT0FBTztFQUNQLDJCQUEyQjtBQUM3Qjs7QUFFQSxpQkFBaUI7QUFDakI7Ozs7Ozs7Ozs7OztFQVlFLE1BQU07RUFDTiwwQkFBMEI7QUFDNUI7O0FBRUEsaUJBQWlCO0FBQ2pCOzs7Ozs7Ozs7Ozs7RUFZRSxnQkFBZ0I7QUFDbEI7O0FBRUEsbUJBQW1CIiwic291cmNlc0NvbnRlbnQiOlsiLyogVGhpcyBmaWxlIGhhcyBjb2RlIGRlcml2ZWQgZnJvbSBMdW1pbm8gQ1NTIGZpbGVzLCBhcyBub3RlZCBiZWxvdy4gVGhlIGxpY2Vuc2UgZm9yIHRoaXMgTHVtaW5vIGNvZGUgaXM6XG5cbkNvcHlyaWdodCAoYykgMjAxOSBQcm9qZWN0IEp1cHl0ZXIgQ29udHJpYnV0b3JzXG5BbGwgcmlnaHRzIHJlc2VydmVkLlxuXG5SZWRpc3RyaWJ1dGlvbiBhbmQgdXNlIGluIHNvdXJjZSBhbmQgYmluYXJ5IGZvcm1zLCB3aXRoIG9yIHdpdGhvdXRcbm1vZGlmaWNhdGlvbiwgYXJlIHBlcm1pdHRlZCBwcm92aWRlZCB0aGF0IHRoZSBmb2xsb3dpbmcgY29uZGl0aW9ucyBhcmUgbWV0OlxuXG4xLiBSZWRpc3RyaWJ1dGlvbnMgb2Ygc291cmNlIGNvZGUgbXVzdCByZXRhaW4gdGhlIGFib3ZlIGNvcHlyaWdodCBub3RpY2UsIHRoaXNcbiAgIGxpc3Qgb2YgY29uZGl0aW9ucyBhbmQgdGhlIGZvbGxvd2luZyBkaXNjbGFpbWVyLlxuXG4yLiBSZWRpc3RyaWJ1dGlvbnMgaW4gYmluYXJ5IGZvcm0gbXVzdCByZXByb2R1Y2UgdGhlIGFib3ZlIGNvcHlyaWdodCBub3RpY2UsXG4gICB0aGlzIGxpc3Qgb2YgY29uZGl0aW9ucyBhbmQgdGhlIGZvbGxvd2luZyBkaXNjbGFpbWVyIGluIHRoZSBkb2N1bWVudGF0aW9uXG4gICBhbmQvb3Igb3RoZXIgbWF0ZXJpYWxzIHByb3ZpZGVkIHdpdGggdGhlIGRpc3RyaWJ1dGlvbi5cblxuMy4gTmVpdGhlciB0aGUgbmFtZSBvZiB0aGUgY29weXJpZ2h0IGhvbGRlciBub3IgdGhlIG5hbWVzIG9mIGl0c1xuICAgY29udHJpYnV0b3JzIG1heSBiZSB1c2VkIHRvIGVuZG9yc2Ugb3IgcHJvbW90ZSBwcm9kdWN0cyBkZXJpdmVkIGZyb21cbiAgIHRoaXMgc29mdHdhcmUgd2l0aG91dCBzcGVjaWZpYyBwcmlvciB3cml0dGVuIHBlcm1pc3Npb24uXG5cblRISVMgU09GVFdBUkUgSVMgUFJPVklERUQgQlkgVEhFIENPUFlSSUdIVCBIT0xERVJTIEFORCBDT05UUklCVVRPUlMgXCJBUyBJU1wiXG5BTkQgQU5ZIEVYUFJFU1MgT1IgSU1QTElFRCBXQVJSQU5USUVTLCBJTkNMVURJTkcsIEJVVCBOT1QgTElNSVRFRCBUTywgVEhFXG5JTVBMSUVEIFdBUlJBTlRJRVMgT0YgTUVSQ0hBTlRBQklMSVRZIEFORCBGSVRORVNTIEZPUiBBIFBBUlRJQ1VMQVIgUFVSUE9TRSBBUkVcbkRJU0NMQUlNRUQuIElOIE5PIEVWRU5UIFNIQUxMIFRIRSBDT1BZUklHSFQgSE9MREVSIE9SIENPTlRSSUJVVE9SUyBCRSBMSUFCTEVcbkZPUiBBTlkgRElSRUNULCBJTkRJUkVDVCwgSU5DSURFTlRBTCwgU1BFQ0lBTCwgRVhFTVBMQVJZLCBPUiBDT05TRVFVRU5USUFMXG5EQU1BR0VTIChJTkNMVURJTkcsIEJVVCBOT1QgTElNSVRFRCBUTywgUFJPQ1VSRU1FTlQgT0YgU1VCU1RJVFVURSBHT09EUyBPUlxuU0VSVklDRVM7IExPU1MgT0YgVVNFLCBEQVRBLCBPUiBQUk9GSVRTOyBPUiBCVVNJTkVTUyBJTlRFUlJVUFRJT04pIEhPV0VWRVJcbkNBVVNFRCBBTkQgT04gQU5ZIFRIRU9SWSBPRiBMSUFCSUxJVFksIFdIRVRIRVIgSU4gQ09OVFJBQ1QsIFNUUklDVCBMSUFCSUxJVFksXG5PUiBUT1JUIChJTkNMVURJTkcgTkVHTElHRU5DRSBPUiBPVEhFUldJU0UpIEFSSVNJTkcgSU4gQU5ZIFdBWSBPVVQgT0YgVEhFIFVTRVxuT0YgVEhJUyBTT0ZUV0FSRSwgRVZFTiBJRiBBRFZJU0VEIE9GIFRIRSBQT1NTSUJJTElUWSBPRiBTVUNIIERBTUFHRS5cblxuXG5Db3B5cmlnaHQgKGMpIDIwMTQtMjAxNywgUGhvc3Bob3JKUyBDb250cmlidXRvcnNcbkFsbCByaWdodHMgcmVzZXJ2ZWQuXG5cblJlZGlzdHJpYnV0aW9uIGFuZCB1c2UgaW4gc291cmNlIGFuZCBiaW5hcnkgZm9ybXMsIHdpdGggb3Igd2l0aG91dFxubW9kaWZpY2F0aW9uLCBhcmUgcGVybWl0dGVkIHByb3ZpZGVkIHRoYXQgdGhlIGZvbGxvd2luZyBjb25kaXRpb25zIGFyZSBtZXQ6XG5cbiogUmVkaXN0cmlidXRpb25zIG9mIHNvdXJjZSBjb2RlIG11c3QgcmV0YWluIHRoZSBhYm92ZSBjb3B5cmlnaHQgbm90aWNlLCB0aGlzXG4gIGxpc3Qgb2YgY29uZGl0aW9ucyBhbmQgdGhlIGZvbGxvd2luZyBkaXNjbGFpbWVyLlxuXG4qIFJlZGlzdHJpYnV0aW9ucyBpbiBiaW5hcnkgZm9ybSBtdXN0IHJlcHJvZHVjZSB0aGUgYWJvdmUgY29weXJpZ2h0IG5vdGljZSxcbiAgdGhpcyBsaXN0IG9mIGNvbmRpdGlvbnMgYW5kIHRoZSBmb2xsb3dpbmcgZGlzY2xhaW1lciBpbiB0aGUgZG9jdW1lbnRhdGlvblxuICBhbmQvb3Igb3RoZXIgbWF0ZXJpYWxzIHByb3ZpZGVkIHdpdGggdGhlIGRpc3RyaWJ1dGlvbi5cblxuKiBOZWl0aGVyIHRoZSBuYW1lIG9mIHRoZSBjb3B5cmlnaHQgaG9sZGVyIG5vciB0aGUgbmFtZXMgb2YgaXRzXG4gIGNvbnRyaWJ1dG9ycyBtYXkgYmUgdXNlZCB0byBlbmRvcnNlIG9yIHByb21vdGUgcHJvZHVjdHMgZGVyaXZlZCBmcm9tXG4gIHRoaXMgc29mdHdhcmUgd2l0aG91dCBzcGVjaWZpYyBwcmlvciB3cml0dGVuIHBlcm1pc3Npb24uXG5cblRISVMgU09GVFdBUkUgSVMgUFJPVklERUQgQlkgVEhFIENPUFlSSUdIVCBIT0xERVJTIEFORCBDT05UUklCVVRPUlMgXCJBUyBJU1wiXG5BTkQgQU5ZIEVYUFJFU1MgT1IgSU1QTElFRCBXQVJSQU5USUVTLCBJTkNMVURJTkcsIEJVVCBOT1QgTElNSVRFRCBUTywgVEhFXG5JTVBMSUVEIFdBUlJBTlRJRVMgT0YgTUVSQ0hBTlRBQklMSVRZIEFORCBGSVRORVNTIEZPUiBBIFBBUlRJQ1VMQVIgUFVSUE9TRSBBUkVcbkRJU0NMQUlNRUQuIElOIE5PIEVWRU5UIFNIQUxMIFRIRSBDT1BZUklHSFQgSE9MREVSIE9SIENPTlRSSUJVVE9SUyBCRSBMSUFCTEVcbkZPUiBBTlkgRElSRUNULCBJTkRJUkVDVCwgSU5DSURFTlRBTCwgU1BFQ0lBTCwgRVhFTVBMQVJZLCBPUiBDT05TRVFVRU5USUFMXG5EQU1BR0VTIChJTkNMVURJTkcsIEJVVCBOT1QgTElNSVRFRCBUTywgUFJPQ1VSRU1FTlQgT0YgU1VCU1RJVFVURSBHT09EUyBPUlxuU0VSVklDRVM7IExPU1MgT0YgVVNFLCBEQVRBLCBPUiBQUk9GSVRTOyBPUiBCVVNJTkVTUyBJTlRFUlJVUFRJT04pIEhPV0VWRVJcbkNBVVNFRCBBTkQgT04gQU5ZIFRIRU9SWSBPRiBMSUFCSUxJVFksIFdIRVRIRVIgSU4gQ09OVFJBQ1QsIFNUUklDVCBMSUFCSUxJVFksXG5PUiBUT1JUIChJTkNMVURJTkcgTkVHTElHRU5DRSBPUiBPVEhFUldJU0UpIEFSSVNJTkcgSU4gQU5ZIFdBWSBPVVQgT0YgVEhFIFVTRVxuT0YgVEhJUyBTT0ZUV0FSRSwgRVZFTiBJRiBBRFZJU0VEIE9GIFRIRSBQT1NTSUJJTElUWSBPRiBTVUNIIERBTUFHRS5cbiovXG5cbi8qXG4gKiBUaGUgZm9sbG93aW5nIHNlY3Rpb24gaXMgZGVyaXZlZCBmcm9tIGh0dHBzOi8vZ2l0aHViLmNvbS9qdXB5dGVybGFiL2x1bWluby9ibG9iLzIzYjlkMDc1ZWJjNWI3M2FiMTQ4YjZlYmZjMjBhZjk3Zjg1NzE0YzQvcGFja2FnZXMvd2lkZ2V0cy9zdHlsZS90YWJiYXIuY3NzIFxuICogV2UndmUgc2NvcGVkIHRoZSBydWxlcyBzbyB0aGF0IHRoZXkgYXJlIGNvbnNpc3RlbnQgd2l0aCBleGFjdGx5IG91ciBjb2RlLlxuICovXG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy53aWRnZXQtdGFiID4gLnAtVGFiQmFyLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi8uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYiA+IC5wLVRhYkJhciwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWIgPiAubG0tVGFiQmFyIHtcbiAgZGlzcGxheTogZmxleDtcbiAgLXdlYmtpdC11c2VyLXNlbGVjdDogbm9uZTtcbiAgLW1vei11c2VyLXNlbGVjdDogbm9uZTtcbiAgLW1zLXVzZXItc2VsZWN0OiBub25lO1xuICB1c2VyLXNlbGVjdDogbm9uZTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLndpZGdldC10YWIgPiAucC1UYWJCYXJbZGF0YS1vcmllbnRhdGlvbj0naG9yaXpvbnRhbCddLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi8uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYiA+IC5wLVRhYkJhcltkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ10sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLmxtLVRhYkJhcltkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ10ge1xuICBmbGV4LWRpcmVjdGlvbjogcm93O1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMud2lkZ2V0LXRhYiA+IC5wLVRhYkJhcltkYXRhLW9yaWVudGF0aW9uPSd2ZXJ0aWNhbCddLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi8uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYiA+IC5wLVRhYkJhcltkYXRhLW9yaWVudGF0aW9uPSd2ZXJ0aWNhbCddLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYiA+IC5sbS1UYWJCYXJbZGF0YS1vcmllbnRhdGlvbj0ndmVydGljYWwnXSB7XG4gIGZsZXgtZGlyZWN0aW9uOiBjb2x1bW47XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy53aWRnZXQtdGFiID4gLnAtVGFiQmFyID4gLnAtVGFiQmFyLWNvbnRlbnQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLy5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLnAtVGFiQmFyID4gLnAtVGFiQmFyLWNvbnRlbnQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLmxtLVRhYkJhciA+IC5sbS1UYWJCYXItY29udGVudCB7XG4gIG1hcmdpbjogMDtcbiAgcGFkZGluZzogMDtcbiAgZGlzcGxheTogZmxleDtcbiAgZmxleDogMSAxIGF1dG87XG4gIGxpc3Qtc3R5bGUtdHlwZTogbm9uZTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLndpZGdldC10YWJcbiAgPiAucC1UYWJCYXJbZGF0YS1vcmllbnRhdGlvbj0naG9yaXpvbnRhbCddXG4gID4gLnAtVGFiQmFyLWNvbnRlbnQsXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiXG4+IC5wLVRhYkJhcltkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ11cbj4gLnAtVGFiQmFyLWNvbnRlbnQsXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYlxuICA+IC5sbS1UYWJCYXJbZGF0YS1vcmllbnRhdGlvbj0naG9yaXpvbnRhbCddXG4gID4gLmxtLVRhYkJhci1jb250ZW50IHtcbiAgZmxleC1kaXJlY3Rpb246IHJvdztcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLndpZGdldC10YWJcbiAgPiAucC1UYWJCYXJbZGF0YS1vcmllbnRhdGlvbj0ndmVydGljYWwnXVxuICA+IC5wLVRhYkJhci1jb250ZW50LFxuLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYlxuPiAucC1UYWJCYXJbZGF0YS1vcmllbnRhdGlvbj0ndmVydGljYWwnXVxuPiAucC1UYWJCYXItY29udGVudCxcbi8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiXG4gID4gLmxtLVRhYkJhcltkYXRhLW9yaWVudGF0aW9uPSd2ZXJ0aWNhbCddXG4gID4gLmxtLVRhYkJhci1jb250ZW50IHtcbiAgZmxleC1kaXJlY3Rpb246IGNvbHVtbjtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLndpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYiwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYiwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWIgPiAubG0tVGFiQmFyIC5sbS1UYWJCYXItdGFiIHtcbiAgZGlzcGxheTogZmxleDtcbiAgZmxleC1kaXJlY3Rpb246IHJvdztcbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcbiAgb3ZlcmZsb3c6IGhpZGRlbjtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLndpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYkljb24sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLyAuanVweXRlci13aWRnZXRzLndpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYkNsb3NlSWNvbiwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYkljb24sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLyAuanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYiA+IC5wLVRhYkJhciAucC1UYWJCYXItdGFiQ2xvc2VJY29uLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYiA+IC5sbS1UYWJCYXIgLmxtLVRhYkJhci10YWJJY29uLFxuLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWIgPiAubG0tVGFiQmFyIC5sbS1UYWJCYXItdGFiQ2xvc2VJY29uIHtcbiAgZmxleDogMCAwIGF1dG87XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy53aWRnZXQtdGFiID4gLnAtVGFiQmFyIC5wLVRhYkJhci10YWJMYWJlbCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYkxhYmVsLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYiA+IC5sbS1UYWJCYXIgLmxtLVRhYkJhci10YWJMYWJlbCB7XG4gIGZsZXg6IDEgMSBhdXRvO1xuICBvdmVyZmxvdzogaGlkZGVuO1xuICB3aGl0ZS1zcGFjZTogbm93cmFwO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMud2lkZ2V0LXRhYiA+IC5wLVRhYkJhciAucC1UYWJCYXItdGFiLnAtbW9kLWhpZGRlbiwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYi5wLW1vZC1oaWRkZW4sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLmxtLVRhYkJhciAubG0tVGFiQmFyLXRhYi5sbS1tb2QtaGlkZGVuIHtcbiAgZGlzcGxheTogbm9uZSAhaW1wb3J0YW50O1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMud2lkZ2V0LXRhYiA+IC5wLVRhYkJhci5wLW1vZC1kcmFnZ2luZyAucC1UYWJCYXItdGFiLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi8uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYiA+IC5wLVRhYkJhci5wLW1vZC1kcmFnZ2luZyAucC1UYWJCYXItdGFiLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYiA+IC5sbS1UYWJCYXIubG0tbW9kLWRyYWdnaW5nIC5sbS1UYWJCYXItdGFiIHtcbiAgcG9zaXRpb246IHJlbGF0aXZlO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMud2lkZ2V0LXRhYlxuICA+IC5wLVRhYkJhci5wLW1vZC1kcmFnZ2luZ1tkYXRhLW9yaWVudGF0aW9uPSdob3Jpem9udGFsJ11cbiAgLnAtVGFiQmFyLXRhYixcbi8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWJcbiAgPiAucC1UYWJCYXIucC1tb2QtZHJhZ2dpbmdbZGF0YS1vcmllbnRhdGlvbj0naG9yaXpvbnRhbCddXG4gIC5wLVRhYkJhci10YWIsXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYlxuICA+IC5sbS1UYWJCYXIubG0tbW9kLWRyYWdnaW5nW2RhdGEtb3JpZW50YXRpb249J2hvcml6b250YWwnXVxuICAubG0tVGFiQmFyLXRhYiB7XG4gIGxlZnQ6IDA7XG4gIHRyYW5zaXRpb246IGxlZnQgMTUwbXMgZWFzZTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLndpZGdldC10YWJcbiAgPiAucC1UYWJCYXIucC1tb2QtZHJhZ2dpbmdbZGF0YS1vcmllbnRhdGlvbj0ndmVydGljYWwnXVxuICAucC1UYWJCYXItdGFiLFxuLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYlxuPiAucC1UYWJCYXIucC1tb2QtZHJhZ2dpbmdbZGF0YS1vcmllbnRhdGlvbj0ndmVydGljYWwnXVxuLnAtVGFiQmFyLXRhYixcbi8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiXG4gID4gLmxtLVRhYkJhci5sbS1tb2QtZHJhZ2dpbmdbZGF0YS1vcmllbnRhdGlvbj0ndmVydGljYWwnXVxuICAubG0tVGFiQmFyLXRhYiB7XG4gIHRvcDogMDtcbiAgdHJhbnNpdGlvbjogdG9wIDE1MG1zIGVhc2U7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy53aWRnZXQtdGFiXG4gID4gLnAtVGFiQmFyLnAtbW9kLWRyYWdnaW5nXG4gIC5wLVRhYkJhci10YWIucC1tb2QtZHJhZ2dpbmcsXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiXG4+IC5wLVRhYkJhci5wLW1vZC1kcmFnZ2luZ1xuLnAtVGFiQmFyLXRhYi5wLW1vZC1kcmFnZ2luZyxcbi8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiXG4gID4gLmxtLVRhYkJhci5sbS1tb2QtZHJhZ2dpbmdcbiAgLmxtLVRhYkJhci10YWIubG0tbW9kLWRyYWdnaW5nIHtcbiAgdHJhbnNpdGlvbjogbm9uZTtcbn1cblxuLyogRW5kIHRhYmJhci5jc3MgKi9cbiJdLCJzb3VyY2VSb290IjoiIn0= */</style><style>/*

The nouislider.css file is autogenerated from nouislider.less, which imports and wraps the nouislider/src/nouislider.less styles.

MIT License

Copyright (c) 2019 Léon Gersen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
/* The .widget-slider class is deprecated */
.widget-slider,
.jupyter-widget-slider {
  /* Functional styling;
 * These styles are required for noUiSlider to function.
 * You don't need to change these rules to apply your design.
 */
  /* Wrapper for all connect elements.
 */
  /* Offset direction
 */
  /* Give origins 0 height/width so they don't interfere with clicking the
 * connect elements.
 */
  /* Slider size and handle placement;
 */
  /* Styling;
 * Giving the connect element a border radius causes issues with using transform: scale
 */
  /* Handles and cursors;
 */
  /* Handle stripes;
 */
  /* Disabled state;
 */
  /* Base;
 *
 */
  /* Values;
 *
 */
  /* Markings;
 *
 */
  /* Horizontal layout;
 *
 */
  /* Vertical layout;
 *
 */
  /* Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */
  /* Custom CSS for nouislider */
}
.widget-slider .noUi-target,
.jupyter-widget-slider .noUi-target,
.widget-slider .noUi-target *,
.jupyter-widget-slider .noUi-target * {
  -webkit-touch-callout: none;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
  -webkit-user-select: none;
  -ms-touch-action: none;
  touch-action: none;
  -ms-user-select: none;
  -moz-user-select: none;
  user-select: none;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
.widget-slider .noUi-target,
.jupyter-widget-slider .noUi-target {
  position: relative;
}
.widget-slider .noUi-base,
.jupyter-widget-slider .noUi-base,
.widget-slider .noUi-connects,
.jupyter-widget-slider .noUi-connects {
  width: 100%;
  height: 100%;
  position: relative;
  z-index: 1;
}
.widget-slider .noUi-connects,
.jupyter-widget-slider .noUi-connects {
  overflow: hidden;
  z-index: 0;
}
.widget-slider .noUi-connect,
.jupyter-widget-slider .noUi-connect,
.widget-slider .noUi-origin,
.jupyter-widget-slider .noUi-origin {
  will-change: transform;
  position: absolute;
  z-index: 1;
  top: 0;
  right: 0;
  -ms-transform-origin: 0 0;
  -webkit-transform-origin: 0 0;
  -webkit-transform-style: preserve-3d;
  transform-origin: 0 0;
  transform-style: flat;
}
.widget-slider .noUi-connect,
.jupyter-widget-slider .noUi-connect {
  height: 100%;
  width: 100%;
}
.widget-slider .noUi-origin,
.jupyter-widget-slider .noUi-origin {
  height: 10%;
  width: 10%;
}
.widget-slider .noUi-txt-dir-rtl.noUi-horizontal .noUi-origin,
.jupyter-widget-slider .noUi-txt-dir-rtl.noUi-horizontal .noUi-origin {
  left: 0;
  right: auto;
}
.widget-slider .noUi-vertical .noUi-origin,
.jupyter-widget-slider .noUi-vertical .noUi-origin {
  width: 0;
}
.widget-slider .noUi-horizontal .noUi-origin,
.jupyter-widget-slider .noUi-horizontal .noUi-origin {
  height: 0;
}
.widget-slider .noUi-handle,
.jupyter-widget-slider .noUi-handle {
  -webkit-backface-visibility: hidden;
  backface-visibility: hidden;
  position: absolute;
}
.widget-slider .noUi-touch-area,
.jupyter-widget-slider .noUi-touch-area {
  height: 100%;
  width: 100%;
}
.widget-slider .noUi-state-tap .noUi-connect,
.jupyter-widget-slider .noUi-state-tap .noUi-connect,
.widget-slider .noUi-state-tap .noUi-origin,
.jupyter-widget-slider .noUi-state-tap .noUi-origin {
  -webkit-transition: transform 0.3s;
  transition: transform 0.3s;
}
.widget-slider .noUi-state-drag *,
.jupyter-widget-slider .noUi-state-drag * {
  cursor: inherit !important;
}
.widget-slider .noUi-horizontal,
.jupyter-widget-slider .noUi-horizontal {
  height: 18px;
}
.widget-slider .noUi-horizontal .noUi-handle,
.jupyter-widget-slider .noUi-horizontal .noUi-handle {
  width: 34px;
  height: 28px;
  right: -17px;
  top: -6px;
}
.widget-slider .noUi-vertical,
.jupyter-widget-slider .noUi-vertical {
  width: 18px;
}
.widget-slider .noUi-vertical .noUi-handle,
.jupyter-widget-slider .noUi-vertical .noUi-handle {
  width: 28px;
  height: 34px;
  right: -6px;
  top: -17px;
}
.widget-slider .noUi-txt-dir-rtl.noUi-horizontal .noUi-handle,
.jupyter-widget-slider .noUi-txt-dir-rtl.noUi-horizontal .noUi-handle {
  left: -17px;
  right: auto;
}
.widget-slider .noUi-target,
.jupyter-widget-slider .noUi-target {
  background: #FAFAFA;
  border-radius: 4px;
  border: 1px solid #D3D3D3;
  box-shadow: inset 0 1px 1px #F0F0F0, 0 3px 6px -5px #BBB;
}
.widget-slider .noUi-connects,
.jupyter-widget-slider .noUi-connects {
  border-radius: 3px;
}
.widget-slider .noUi-connect,
.jupyter-widget-slider .noUi-connect {
  background: #3FB8AF;
}
.widget-slider .noUi-draggable,
.jupyter-widget-slider .noUi-draggable {
  cursor: ew-resize;
}
.widget-slider .noUi-vertical .noUi-draggable,
.jupyter-widget-slider .noUi-vertical .noUi-draggable {
  cursor: ns-resize;
}
.widget-slider .noUi-handle,
.jupyter-widget-slider .noUi-handle {
  border: 1px solid #D9D9D9;
  border-radius: 3px;
  background: #FFF;
  cursor: default;
  box-shadow: inset 0 0 1px #FFF, inset 0 1px 7px #EBEBEB, 0 3px 6px -3px #BBB;
}
.widget-slider .noUi-active,
.jupyter-widget-slider .noUi-active {
  box-shadow: inset 0 0 1px #FFF, inset 0 1px 7px #DDD, 0 3px 6px -3px #BBB;
}
.widget-slider .noUi-handle:before,
.jupyter-widget-slider .noUi-handle:before,
.widget-slider .noUi-handle:after,
.jupyter-widget-slider .noUi-handle:after {
  content: "";
  display: block;
  position: absolute;
  height: 14px;
  width: 1px;
  background: #E8E7E6;
  left: 14px;
  top: 6px;
}
.widget-slider .noUi-handle:after,
.jupyter-widget-slider .noUi-handle:after {
  left: 17px;
}
.widget-slider .noUi-vertical .noUi-handle:before,
.jupyter-widget-slider .noUi-vertical .noUi-handle:before,
.widget-slider .noUi-vertical .noUi-handle:after,
.jupyter-widget-slider .noUi-vertical .noUi-handle:after {
  width: 14px;
  height: 1px;
  left: 6px;
  top: 14px;
}
.widget-slider .noUi-vertical .noUi-handle:after,
.jupyter-widget-slider .noUi-vertical .noUi-handle:after {
  top: 17px;
}
.widget-slider [disabled] .noUi-connect,
.jupyter-widget-slider [disabled] .noUi-connect {
  background: #B8B8B8;
}
.widget-slider [disabled].noUi-target,
.jupyter-widget-slider [disabled].noUi-target,
.widget-slider [disabled].noUi-handle,
.jupyter-widget-slider [disabled].noUi-handle,
.widget-slider [disabled] .noUi-handle,
.jupyter-widget-slider [disabled] .noUi-handle {
  cursor: not-allowed;
}
.widget-slider .noUi-pips,
.jupyter-widget-slider .noUi-pips,
.widget-slider .noUi-pips *,
.jupyter-widget-slider .noUi-pips * {
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
.widget-slider .noUi-pips,
.jupyter-widget-slider .noUi-pips {
  position: absolute;
  color: #999;
}
.widget-slider .noUi-value,
.jupyter-widget-slider .noUi-value {
  position: absolute;
  white-space: nowrap;
  text-align: center;
}
.widget-slider .noUi-value-sub,
.jupyter-widget-slider .noUi-value-sub {
  color: #ccc;
  font-size: 10px;
}
.widget-slider .noUi-marker,
.jupyter-widget-slider .noUi-marker {
  position: absolute;
  background: #CCC;
}
.widget-slider .noUi-marker-sub,
.jupyter-widget-slider .noUi-marker-sub {
  background: #AAA;
}
.widget-slider .noUi-marker-large,
.jupyter-widget-slider .noUi-marker-large {
  background: #AAA;
}
.widget-slider .noUi-pips-horizontal,
.jupyter-widget-slider .noUi-pips-horizontal {
  padding: 10px 0;
  height: 80px;
  top: 100%;
  left: 0;
  width: 100%;
}
.widget-slider .noUi-value-horizontal,
.jupyter-widget-slider .noUi-value-horizontal {
  -webkit-transform: translate(-50%, 50%);
  transform: translate(-50%, 50%);
}
.noUi-rtl .widget-slider .noUi-value-horizontal,
.noUi-rtl .jupyter-widget-slider .noUi-value-horizontal {
  -webkit-transform: translate(50%, 50%);
  transform: translate(50%, 50%);
}
.widget-slider .noUi-marker-horizontal.noUi-marker,
.jupyter-widget-slider .noUi-marker-horizontal.noUi-marker {
  margin-left: -1px;
  width: 2px;
  height: 5px;
}
.widget-slider .noUi-marker-horizontal.noUi-marker-sub,
.jupyter-widget-slider .noUi-marker-horizontal.noUi-marker-sub {
  height: 10px;
}
.widget-slider .noUi-marker-horizontal.noUi-marker-large,
.jupyter-widget-slider .noUi-marker-horizontal.noUi-marker-large {
  height: 15px;
}
.widget-slider .noUi-pips-vertical,
.jupyter-widget-slider .noUi-pips-vertical {
  padding: 0 10px;
  height: 100%;
  top: 0;
  left: 100%;
}
.widget-slider .noUi-value-vertical,
.jupyter-widget-slider .noUi-value-vertical {
  -webkit-transform: translate(0, -50%);
  transform: translate(0, -50%);
  padding-left: 25px;
}
.noUi-rtl .widget-slider .noUi-value-vertical,
.noUi-rtl .jupyter-widget-slider .noUi-value-vertical {
  -webkit-transform: translate(0, 50%);
  transform: translate(0, 50%);
}
.widget-slider .noUi-marker-vertical.noUi-marker,
.jupyter-widget-slider .noUi-marker-vertical.noUi-marker {
  width: 5px;
  height: 2px;
  margin-top: -1px;
}
.widget-slider .noUi-marker-vertical.noUi-marker-sub,
.jupyter-widget-slider .noUi-marker-vertical.noUi-marker-sub {
  width: 10px;
}
.widget-slider .noUi-marker-vertical.noUi-marker-large,
.jupyter-widget-slider .noUi-marker-vertical.noUi-marker-large {
  width: 15px;
}
.widget-slider .noUi-tooltip,
.jupyter-widget-slider .noUi-tooltip {
  display: block;
  position: absolute;
  border: 1px solid #D9D9D9;
  border-radius: 3px;
  background: #fff;
  color: #000;
  padding: 5px;
  text-align: center;
  white-space: nowrap;
}
.widget-slider .noUi-horizontal .noUi-tooltip,
.jupyter-widget-slider .noUi-horizontal .noUi-tooltip {
  -webkit-transform: translate(-50%, 0);
  transform: translate(-50%, 0);
  left: 50%;
  bottom: 120%;
}
.widget-slider .noUi-vertical .noUi-tooltip,
.jupyter-widget-slider .noUi-vertical .noUi-tooltip {
  -webkit-transform: translate(0, -50%);
  transform: translate(0, -50%);
  top: 50%;
  right: 120%;
}
.widget-slider .noUi-horizontal .noUi-origin > .noUi-tooltip,
.jupyter-widget-slider .noUi-horizontal .noUi-origin > .noUi-tooltip {
  -webkit-transform: translate(50%, 0);
  transform: translate(50%, 0);
  left: auto;
  bottom: 10px;
}
.widget-slider .noUi-vertical .noUi-origin > .noUi-tooltip,
.jupyter-widget-slider .noUi-vertical .noUi-origin > .noUi-tooltip {
  -webkit-transform: translate(0, -18px);
  transform: translate(0, -18px);
  top: auto;
  right: 28px;
}
.widget-slider .noUi-connect,
.jupyter-widget-slider .noUi-connect {
  background: #2196f3;
}
.widget-slider .noUi-horizontal,
.jupyter-widget-slider .noUi-horizontal {
  height: var(--jp-widgets-slider-track-thickness);
}
.widget-slider .noUi-vertical,
.jupyter-widget-slider .noUi-vertical {
  width: var(--jp-widgets-slider-track-thickness);
  height: 100%;
}
.widget-slider .noUi-horizontal .noUi-handle,
.jupyter-widget-slider .noUi-horizontal .noUi-handle {
  width: var(--jp-widgets-slider-handle-size);
  height: var(--jp-widgets-slider-handle-size);
  border-radius: 50%;
  top: calc((var(--jp-widgets-slider-track-thickness) - var(--jp-widgets-slider-handle-size)) / 2);
  right: calc(var(--jp-widgets-slider-handle-size) / -2);
}
.widget-slider .noUi-vertical .noUi-handle,
.jupyter-widget-slider .noUi-vertical .noUi-handle {
  height: var(--jp-widgets-slider-handle-size);
  width: var(--jp-widgets-slider-handle-size);
  border-radius: 50%;
  right: calc((var(--jp-widgets-slider-handle-size) - var(--jp-widgets-slider-track-thickness)) / -2);
  top: calc(var(--jp-widgets-slider-handle-size) / -2);
}
.widget-slider .noUi-handle:after,
.jupyter-widget-slider .noUi-handle:after {
  content: none;
}
.widget-slider .noUi-handle:before,
.jupyter-widget-slider .noUi-handle:before {
  content: none;
}
.widget-slider .noUi-target,
.jupyter-widget-slider .noUi-target {
  background: #fafafa;
  border-radius: 4px;
  border: 1px;
  /* box-shadow: inset 0 1px 1px #F0F0F0, 0 3px 6px -5px #BBB; */
}
.widget-slider .ui-slider,
.jupyter-widget-slider .ui-slider {
  border: var(--jp-widgets-slider-border-width) solid var(--jp-layout-color3);
  background: var(--jp-layout-color3);
  box-sizing: border-box;
  position: relative;
  border-radius: 0px;
}
.widget-slider .noUi-handle,
.jupyter-widget-slider .noUi-handle {
  width: var(--jp-widgets-slider-handle-size);
  border: 1px solid #d9d9d9;
  border-radius: 3px;
  background: #fff;
  cursor: default;
  box-shadow: none;
  outline: none;
}
.widget-slider .noUi-target:not([disabled]) .noUi-handle:hover,
.jupyter-widget-slider .noUi-target:not([disabled]) .noUi-handle:hover,
.widget-slider .noUi-target:not([disabled]) .noUi-handle:focus,
.jupyter-widget-slider .noUi-target:not([disabled]) .noUi-handle:focus {
  background-color: var(--jp-widgets-slider-active-handle-color);
  border: var(--jp-widgets-slider-border-width) solid var(--jp-widgets-slider-active-handle-color);
}
.widget-slider [disabled].noUi-target,
.jupyter-widget-slider [disabled].noUi-target {
  opacity: 0.35;
}
.widget-slider .noUi-connects,
.jupyter-widget-slider .noUi-connects {
  overflow: visible;
  z-index: 0;
  background: var(--jp-layout-color3);
}
.widget-slider .noUi-vertical .noUi-connect,
.jupyter-widget-slider .noUi-vertical .noUi-connect {
  width: calc(100% + 2px);
  right: -1px;
}
.widget-slider .noUi-horizontal .noUi-connect,
.jupyter-widget-slider .noUi-horizontal .noUi-connect {
  height: calc(100% + 2px);
  top: -1px;
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL3BhY2thZ2VzL2NvbnRyb2xzL2Nzcy9ub3Vpc2xpZGVyLmNzcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7OztDQWFDO0FBQ0QsMkNBQTJDO0FBQzNDOztFQUVFOzs7RUFHQTtFQUNBO0VBQ0E7RUFDQTtFQUNBO0VBQ0E7O0VBRUE7RUFDQTtFQUNBO0VBQ0E7O0VBRUE7RUFDQTtFQUNBO0VBQ0E7RUFDQTtFQUNBO0VBQ0E7RUFDQTs7RUFFQTtFQUNBOztFQUVBO0VBQ0E7O0VBRUE7RUFDQTs7RUFFQTtFQUNBOztFQUVBO0VBQ0E7O0VBRUE7RUFDQSw4QkFBOEI7QUFDaEM7QUFDQTs7OztFQUlFLDJCQUEyQjtFQUMzQiw2Q0FBNkM7RUFDN0MseUJBQXlCO0VBQ3pCLHNCQUFzQjtFQUN0QixrQkFBa0I7RUFDbEIscUJBQXFCO0VBQ3JCLHNCQUFzQjtFQUN0QixpQkFBaUI7RUFDakIsMkJBQTJCO0VBQzNCLHNCQUFzQjtBQUN4QjtBQUNBOztFQUVFLGtCQUFrQjtBQUNwQjtBQUNBOzs7O0VBSUUsV0FBVztFQUNYLFlBQVk7RUFDWixrQkFBa0I7RUFDbEIsVUFBVTtBQUNaO0FBQ0E7O0VBRUUsZ0JBQWdCO0VBQ2hCLFVBQVU7QUFDWjtBQUNBOzs7O0VBSUUsc0JBQXNCO0VBQ3RCLGtCQUFrQjtFQUNsQixVQUFVO0VBQ1YsTUFBTTtFQUNOLFFBQVE7RUFDUix5QkFBeUI7RUFDekIsNkJBQTZCO0VBQzdCLG9DQUFvQztFQUNwQyxxQkFBcUI7RUFDckIscUJBQXFCO0FBQ3ZCO0FBQ0E7O0VBRUUsWUFBWTtFQUNaLFdBQVc7QUFDYjtBQUNBOztFQUVFLFdBQVc7RUFDWCxVQUFVO0FBQ1o7QUFDQTs7RUFFRSxPQUFPO0VBQ1AsV0FBVztBQUNiO0FBQ0E7O0VBRUUsUUFBUTtBQUNWO0FBQ0E7O0VBRUUsU0FBUztBQUNYO0FBQ0E7O0VBRUUsbUNBQW1DO0VBQ25DLDJCQUEyQjtFQUMzQixrQkFBa0I7QUFDcEI7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Q2I7QUFDQTs7RUFFRSxtQkFBbUI7QUFDckI7QUFDQTs7RUFFRSxnREFBZ0Q7QUFDbEQ7QUFDQTs7RUFFRSwrQ0FBK0M7RUFDL0MsWUFBWTtBQUNkO0FBQ0E7O0VBRUUsMkNBQTJDO0VBQzNDLDRDQUE0QztFQUM1QyxrQkFBa0I7RUFDbEIsZ0dBQWdHO0VBQ2hHLHNEQUFzRDtBQUN4RDtBQUNBOztFQUVFLDRDQUE0QztFQUM1QywyQ0FBMkM7RUFDM0Msa0JBQWtCO0VBQ2xCLG1HQUFtRztFQUNuRyxvREFBb0Q7QUFDdEQ7QUFDQTs7RUFFRSxhQUFhO0FBQ2Y7QUFDQTs7RUFFRSxhQUFhO0FBQ2Y7QUFDQTs7RUFFRSxtQkFBbUI7RUFDbkIsa0JBQWtCO0VBQ2xCLFdBQVc7RUFDWCw4REFBOEQ7QUFDaEU7QUFDQTs7RUFFRSwyRUFBMkU7RUFDM0UsbUNBQW1DO0VBQ25DLHNCQUFzQjtFQUN0QixrQkFBa0I7RUFDbEIsa0JBQWtCO0FBQ3BCO0FBQ0E7O0VBRUUsMkNBQTJDO0VBQzNDLHlCQUF5QjtFQUN6QixrQkFBa0I7RUFDbEIsZ0JBQWdCO0VBQ2hCLGVBQWU7RUFDZixnQkFBZ0I7RUFDaEIsYUFBYTtBQUNmO0FBQ0E7Ozs7RUFJRSw4REFBOEQ7RUFDOUQsZ0dBQWdHO0FBQ2xHO0FBQ0E7O0VBRUUsYUFBYTtBQUNmO0FBQ0E7O0VBRUUsaUJBQWlCO0VBQ2pCLFVBQVU7RUFDVixtQ0FBbUM7QUFDckM7QUFDQTs7RUFFRSx1QkFBdUI7RUFDdkIsV0FBVztBQUNiO0FBQ0E7O0VBRUUsd0JBQXdCO0VBQ3hCLFNBQVM7QUFDWCIsInNvdXJjZXNDb250ZW50IjpbIi8qXG5cblRoZSBub3Vpc2xpZGVyLmNzcyBmaWxlIGlzIGF1dG9nZW5lcmF0ZWQgZnJvbSBub3Vpc2xpZGVyLmxlc3MsIHdoaWNoIGltcG9ydHMgYW5kIHdyYXBzIHRoZSBub3Vpc2xpZGVyL3NyYy9ub3Vpc2xpZGVyLmxlc3Mgc3R5bGVzLlxuXG5NSVQgTGljZW5zZVxuXG5Db3B5cmlnaHQgKGMpIDIwMTkgTMOpb24gR2Vyc2VuXG5cblBlcm1pc3Npb24gaXMgaGVyZWJ5IGdyYW50ZWQsIGZyZWUgb2YgY2hhcmdlLCB0byBhbnkgcGVyc29uIG9idGFpbmluZyBhIGNvcHkgb2YgdGhpcyBzb2Z0d2FyZSBhbmQgYXNzb2NpYXRlZCBkb2N1bWVudGF0aW9uIGZpbGVzICh0aGUgXCJTb2Z0d2FyZVwiKSwgdG8gZGVhbCBpbiB0aGUgU29mdHdhcmUgd2l0aG91dCByZXN0cmljdGlvbiwgaW5jbHVkaW5nIHdpdGhvdXQgbGltaXRhdGlvbiB0aGUgcmlnaHRzIHRvIHVzZSwgY29weSwgbW9kaWZ5LCBtZXJnZSwgcHVibGlzaCwgZGlzdHJpYnV0ZSwgc3VibGljZW5zZSwgYW5kL29yIHNlbGwgY29waWVzIG9mIHRoZSBTb2Z0d2FyZSwgYW5kIHRvIHBlcm1pdCBwZXJzb25zIHRvIHdob20gdGhlIFNvZnR3YXJlIGlzIGZ1cm5pc2hlZCB0byBkbyBzbywgc3ViamVjdCB0byB0aGUgZm9sbG93aW5nIGNvbmRpdGlvbnM6XG5cblRoZSBhYm92ZSBjb3B5cmlnaHQgbm90aWNlIGFuZCB0aGlzIHBlcm1pc3Npb24gbm90aWNlIHNoYWxsIGJlIGluY2x1ZGVkIGluIGFsbCBjb3BpZXMgb3Igc3Vic3RhbnRpYWwgcG9ydGlvbnMgb2YgdGhlIFNvZnR3YXJlLlxuXG5USEUgU09GVFdBUkUgSVMgUFJPVklERUQgXCJBUyBJU1wiLCBXSVRIT1VUIFdBUlJBTlRZIE9GIEFOWSBLSU5ELCBFWFBSRVNTIE9SIElNUExJRUQsIElOQ0xVRElORyBCVVQgTk9UIExJTUlURUQgVE8gVEhFIFdBUlJBTlRJRVMgT0YgTUVSQ0hBTlRBQklMSVRZLCBGSVRORVNTIEZPUiBBIFBBUlRJQ1VMQVIgUFVSUE9TRSBBTkQgTk9OSU5GUklOR0VNRU5ULiBJTiBOTyBFVkVOVCBTSEFMTCBUSEUgQVVUSE9SUyBPUiBDT1BZUklHSFQgSE9MREVSUyBCRSBMSUFCTEUgRk9SIEFOWSBDTEFJTSwgREFNQUdFUyBPUiBPVEhFUiBMSUFCSUxJVFksIFdIRVRIRVIgSU4gQU4gQUNUSU9OIE9GIENPTlRSQUNULCBUT1JUIE9SIE9USEVSV0lTRSwgQVJJU0lORyBGUk9NLCBPVVQgT0YgT1IgSU4gQ09OTkVDVElPTiBXSVRIIFRIRSBTT0ZUV0FSRSBPUiBUSEUgVVNFIE9SIE9USEVSIERFQUxJTkdTIElOIFRIRSBTT0ZUV0FSRS5cbiovXG4vKiBUaGUgLndpZGdldC1zbGlkZXIgY2xhc3MgaXMgZGVwcmVjYXRlZCAqL1xuLndpZGdldC1zbGlkZXIsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIHtcbiAgLyogRnVuY3Rpb25hbCBzdHlsaW5nO1xuICogVGhlc2Ugc3R5bGVzIGFyZSByZXF1aXJlZCBmb3Igbm9VaVNsaWRlciB0byBmdW5jdGlvbi5cbiAqIFlvdSBkb24ndCBuZWVkIHRvIGNoYW5nZSB0aGVzZSBydWxlcyB0byBhcHBseSB5b3VyIGRlc2lnbi5cbiAqL1xuICAvKiBXcmFwcGVyIGZvciBhbGwgY29ubmVjdCBlbGVtZW50cy5cbiAqL1xuICAvKiBPZmZzZXQgZGlyZWN0aW9uXG4gKi9cbiAgLyogR2l2ZSBvcmlnaW5zIDAgaGVpZ2h0L3dpZHRoIHNvIHRoZXkgZG9uJ3QgaW50ZXJmZXJlIHdpdGggY2xpY2tpbmcgdGhlXG4gKiBjb25uZWN0IGVsZW1lbnRzLlxuICovXG4gIC8qIFNsaWRlciBzaXplIGFuZCBoYW5kbGUgcGxhY2VtZW50O1xuICovXG4gIC8qIFN0eWxpbmc7XG4gKiBHaXZpbmcgdGhlIGNvbm5lY3QgZWxlbWVudCBhIGJvcmRlciByYWRpdXMgY2F1c2VzIGlzc3VlcyB3aXRoIHVzaW5nIHRyYW5zZm9ybTogc2NhbGVcbiAqL1xuICAvKiBIYW5kbGVzIGFuZCBjdXJzb3JzO1xuICovXG4gIC8qIEhhbmRsZSBzdHJpcGVzO1xuICovXG4gIC8qIERpc2FibGVkIHN0YXRlO1xuICovXG4gIC8qIEJhc2U7XG4gKlxuICovXG4gIC8qIFZhbHVlcztcbiAqXG4gKi9cbiAgLyogTWFya2luZ3M7XG4gKlxuICovXG4gIC8qIEhvcml6b250YWwgbGF5b3V0O1xuICpcbiAqL1xuICAvKiBWZXJ0aWNhbCBsYXlvdXQ7XG4gKlxuICovXG4gIC8qIENvcHlyaWdodCAoYykgSnVweXRlciBEZXZlbG9wbWVudCBUZWFtLlxuICogRGlzdHJpYnV0ZWQgdW5kZXIgdGhlIHRlcm1zIG9mIHRoZSBNb2RpZmllZCBCU0QgTGljZW5zZS5cbiAqL1xuICAvKiBDdXN0b20gQ1NTIGZvciBub3Vpc2xpZGVyICovXG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS10YXJnZXQsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXRhcmdldCxcbi53aWRnZXQtc2xpZGVyIC5ub1VpLXRhcmdldCAqLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS10YXJnZXQgKiB7XG4gIC13ZWJraXQtdG91Y2gtY2FsbG91dDogbm9uZTtcbiAgLXdlYmtpdC10YXAtaGlnaGxpZ2h0LWNvbG9yOiByZ2JhKDAsIDAsIDAsIDApO1xuICAtd2Via2l0LXVzZXItc2VsZWN0OiBub25lO1xuICAtbXMtdG91Y2gtYWN0aW9uOiBub25lO1xuICB0b3VjaC1hY3Rpb246IG5vbmU7XG4gIC1tcy11c2VyLXNlbGVjdDogbm9uZTtcbiAgLW1vei11c2VyLXNlbGVjdDogbm9uZTtcbiAgdXNlci1zZWxlY3Q6IG5vbmU7XG4gIC1tb3otYm94LXNpemluZzogYm9yZGVyLWJveDtcbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLXRhcmdldCxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktdGFyZ2V0IHtcbiAgcG9zaXRpb246IHJlbGF0aXZlO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktYmFzZSxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktYmFzZSxcbi53aWRnZXQtc2xpZGVyIC5ub1VpLWNvbm5lY3RzLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1jb25uZWN0cyB7XG4gIHdpZHRoOiAxMDAlO1xuICBoZWlnaHQ6IDEwMCU7XG4gIHBvc2l0aW9uOiByZWxhdGl2ZTtcbiAgei1pbmRleDogMTtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLWNvbm5lY3RzLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1jb25uZWN0cyB7XG4gIG92ZXJmbG93OiBoaWRkZW47XG4gIHotaW5kZXg6IDA7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS1jb25uZWN0LFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1jb25uZWN0LFxuLndpZGdldC1zbGlkZXIgLm5vVWktb3JpZ2luLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1vcmlnaW4ge1xuICB3aWxsLWNoYW5nZTogdHJhbnNmb3JtO1xuICBwb3NpdGlvbjogYWJzb2x1dGU7XG4gIHotaW5kZXg6IDE7XG4gIHRvcDogMDtcbiAgcmlnaHQ6IDA7XG4gIC1tcy10cmFuc2Zvcm0tb3JpZ2luOiAwIDA7XG4gIC13ZWJraXQtdHJhbnNmb3JtLW9yaWdpbjogMCAwO1xuICAtd2Via2l0LXRyYW5zZm9ybS1zdHlsZTogcHJlc2VydmUtM2Q7XG4gIHRyYW5zZm9ybS1vcmlnaW46IDAgMDtcbiAgdHJhbnNmb3JtLXN0eWxlOiBmbGF0O1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktY29ubmVjdCxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktY29ubmVjdCB7XG4gIGhlaWdodDogMTAwJTtcbiAgd2lkdGg6IDEwMCU7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS1vcmlnaW4sXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLW9yaWdpbiB7XG4gIGhlaWdodDogMTAlO1xuICB3aWR0aDogMTAlO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktdHh0LWRpci1ydGwubm9VaS1ob3Jpem9udGFsIC5ub1VpLW9yaWdpbixcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktdHh0LWRpci1ydGwubm9VaS1ob3Jpem9udGFsIC5ub1VpLW9yaWdpbiB7XG4gIGxlZnQ6IDA7XG4gIHJpZ2h0OiBhdXRvO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktdmVydGljYWwgLm5vVWktb3JpZ2luLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS12ZXJ0aWNhbCAubm9VaS1vcmlnaW4ge1xuICB3aWR0aDogMDtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLWhvcml6b250YWwgLm5vVWktb3JpZ2luLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1ob3Jpem9udGFsIC5ub1VpLW9yaWdpbiB7XG4gIGhlaWdodDogMDtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLWhhbmRsZSxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktaGFuZGxlIHtcbiAgLXdlYmtpdC1iYWNrZmFjZS12aXNpYmlsaXR5OiBoaWRkZW47XG4gIGJhY2tmYWNlLXZpc2liaWxpdHk6IGhpZGRlbjtcbiAgcG9zaXRpb246IGFic29sdXRlO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktdG91Y2gtYXJlYSxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktdG91Y2gtYXJlYSB7XG4gIGhlaWdodDogMTAwJTtcbiAgd2lkdGg6IDEwMCU7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS1zdGF0ZS10YXAgLm5vVWktY29ubmVjdCxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktc3RhdGUtdGFwIC5ub1VpLWNvbm5lY3QsXG4ud2lkZ2V0LXNsaWRlciAubm9VaS1zdGF0ZS10YXAgLm5vVWktb3JpZ2luLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1zdGF0ZS10YXAgLm5vVWktb3JpZ2luIHtcbiAgLXdlYmtpdC10cmFuc2l0aW9uOiB0cmFuc2Zvcm0gMC4zcztcbiAgdHJhbnNpdGlvbjogdHJhbnNmb3JtIDAuM3M7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS1zdGF0ZS1kcmFnICosXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXN0YXRlLWRyYWcgKiB7XG4gIGN1cnNvcjogaW5oZXJpdCAhaW1wb3J0YW50O1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktaG9yaXpvbnRhbCxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktaG9yaXpvbnRhbCB7XG4gIGhlaWdodDogMThweDtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLWhvcml6b250YWwgLm5vVWktaGFuZGxlLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1ob3Jpem9udGFsIC5ub1VpLWhhbmRsZSB7XG4gIHdpZHRoOiAzNHB4O1xuICBoZWlnaHQ6IDI4cHg7XG4gIHJpZ2h0OiAtMTdweDtcbiAgdG9wOiAtNnB4O1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktdmVydGljYWwsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXZlcnRpY2FsIHtcbiAgd2lkdGg6IDE4cHg7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS12ZXJ0aWNhbCAubm9VaS1oYW5kbGUsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXZlcnRpY2FsIC5ub1VpLWhhbmRsZSB7XG4gIHdpZHRoOiAyOHB4O1xuICBoZWlnaHQ6IDM0cHg7XG4gIHJpZ2h0OiAtNnB4O1xuICB0b3A6IC0xN3B4O1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktdHh0LWRpci1ydGwubm9VaS1ob3Jpem9udGFsIC5ub1VpLWhhbmRsZSxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktdHh0LWRpci1ydGwubm9VaS1ob3Jpem9udGFsIC5ub1VpLWhhbmRsZSB7XG4gIGxlZnQ6IC0xN3B4O1xuICByaWdodDogYXV0bztcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLXRhcmdldCxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktdGFyZ2V0IHtcbiAgYmFja2dyb3VuZDogI0ZBRkFGQTtcbiAgYm9yZGVyLXJhZGl1czogNHB4O1xuICBib3JkZXI6IDFweCBzb2xpZCAjRDNEM0QzO1xuICBib3gtc2hhZG93OiBpbnNldCAwIDFweCAxcHggI0YwRjBGMCwgMCAzcHggNnB4IC01cHggI0JCQjtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLWNvbm5lY3RzLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1jb25uZWN0cyB7XG4gIGJvcmRlci1yYWRpdXM6IDNweDtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLWNvbm5lY3QsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLWNvbm5lY3Qge1xuICBiYWNrZ3JvdW5kOiAjM0ZCOEFGO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktZHJhZ2dhYmxlLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1kcmFnZ2FibGUge1xuICBjdXJzb3I6IGV3LXJlc2l6ZTtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLXZlcnRpY2FsIC5ub1VpLWRyYWdnYWJsZSxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktdmVydGljYWwgLm5vVWktZHJhZ2dhYmxlIHtcbiAgY3Vyc29yOiBucy1yZXNpemU7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS1oYW5kbGUsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLWhhbmRsZSB7XG4gIGJvcmRlcjogMXB4IHNvbGlkICNEOUQ5RDk7XG4gIGJvcmRlci1yYWRpdXM6IDNweDtcbiAgYmFja2dyb3VuZDogI0ZGRjtcbiAgY3Vyc29yOiBkZWZhdWx0O1xuICBib3gtc2hhZG93OiBpbnNldCAwIDAgMXB4ICNGRkYsIGluc2V0IDAgMXB4IDdweCAjRUJFQkVCLCAwIDNweCA2cHggLTNweCAjQkJCO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktYWN0aXZlLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1hY3RpdmUge1xuICBib3gtc2hhZG93OiBpbnNldCAwIDAgMXB4ICNGRkYsIGluc2V0IDAgMXB4IDdweCAjRERELCAwIDNweCA2cHggLTNweCAjQkJCO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktaGFuZGxlOmJlZm9yZSxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktaGFuZGxlOmJlZm9yZSxcbi53aWRnZXQtc2xpZGVyIC5ub1VpLWhhbmRsZTphZnRlcixcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktaGFuZGxlOmFmdGVyIHtcbiAgY29udGVudDogXCJcIjtcbiAgZGlzcGxheTogYmxvY2s7XG4gIHBvc2l0aW9uOiBhYnNvbHV0ZTtcbiAgaGVpZ2h0OiAxNHB4O1xuICB3aWR0aDogMXB4O1xuICBiYWNrZ3JvdW5kOiAjRThFN0U2O1xuICBsZWZ0OiAxNHB4O1xuICB0b3A6IDZweDtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLWhhbmRsZTphZnRlcixcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktaGFuZGxlOmFmdGVyIHtcbiAgbGVmdDogMTdweDtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLXZlcnRpY2FsIC5ub1VpLWhhbmRsZTpiZWZvcmUsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXZlcnRpY2FsIC5ub1VpLWhhbmRsZTpiZWZvcmUsXG4ud2lkZ2V0LXNsaWRlciAubm9VaS12ZXJ0aWNhbCAubm9VaS1oYW5kbGU6YWZ0ZXIsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXZlcnRpY2FsIC5ub1VpLWhhbmRsZTphZnRlciB7XG4gIHdpZHRoOiAxNHB4O1xuICBoZWlnaHQ6IDFweDtcbiAgbGVmdDogNnB4O1xuICB0b3A6IDE0cHg7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS12ZXJ0aWNhbCAubm9VaS1oYW5kbGU6YWZ0ZXIsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXZlcnRpY2FsIC5ub1VpLWhhbmRsZTphZnRlciB7XG4gIHRvcDogMTdweDtcbn1cbi53aWRnZXQtc2xpZGVyIFtkaXNhYmxlZF0gLm5vVWktY29ubmVjdCxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgW2Rpc2FibGVkXSAubm9VaS1jb25uZWN0IHtcbiAgYmFja2dyb3VuZDogI0I4QjhCODtcbn1cbi53aWRnZXQtc2xpZGVyIFtkaXNhYmxlZF0ubm9VaS10YXJnZXQsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIFtkaXNhYmxlZF0ubm9VaS10YXJnZXQsXG4ud2lkZ2V0LXNsaWRlciBbZGlzYWJsZWRdLm5vVWktaGFuZGxlLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciBbZGlzYWJsZWRdLm5vVWktaGFuZGxlLFxuLndpZGdldC1zbGlkZXIgW2Rpc2FibGVkXSAubm9VaS1oYW5kbGUsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIFtkaXNhYmxlZF0gLm5vVWktaGFuZGxlIHtcbiAgY3Vyc29yOiBub3QtYWxsb3dlZDtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLXBpcHMsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXBpcHMsXG4ud2lkZ2V0LXNsaWRlciAubm9VaS1waXBzICosXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXBpcHMgKiB7XG4gIC1tb3otYm94LXNpemluZzogYm9yZGVyLWJveDtcbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLXBpcHMsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXBpcHMge1xuICBwb3NpdGlvbjogYWJzb2x1dGU7XG4gIGNvbG9yOiAjOTk5O1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktdmFsdWUsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXZhbHVlIHtcbiAgcG9zaXRpb246IGFic29sdXRlO1xuICB3aGl0ZS1zcGFjZTogbm93cmFwO1xuICB0ZXh0LWFsaWduOiBjZW50ZXI7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS12YWx1ZS1zdWIsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXZhbHVlLXN1YiB7XG4gIGNvbG9yOiAjY2NjO1xuICBmb250LXNpemU6IDEwcHg7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS1tYXJrZXIsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLW1hcmtlciB7XG4gIHBvc2l0aW9uOiBhYnNvbHV0ZTtcbiAgYmFja2dyb3VuZDogI0NDQztcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLW1hcmtlci1zdWIsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLW1hcmtlci1zdWIge1xuICBiYWNrZ3JvdW5kOiAjQUFBO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktbWFya2VyLWxhcmdlLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1tYXJrZXItbGFyZ2Uge1xuICBiYWNrZ3JvdW5kOiAjQUFBO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktcGlwcy1ob3Jpem9udGFsLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1waXBzLWhvcml6b250YWwge1xuICBwYWRkaW5nOiAxMHB4IDA7XG4gIGhlaWdodDogODBweDtcbiAgdG9wOiAxMDAlO1xuICBsZWZ0OiAwO1xuICB3aWR0aDogMTAwJTtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLXZhbHVlLWhvcml6b250YWwsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXZhbHVlLWhvcml6b250YWwge1xuICAtd2Via2l0LXRyYW5zZm9ybTogdHJhbnNsYXRlKC01MCUsIDUwJSk7XG4gIHRyYW5zZm9ybTogdHJhbnNsYXRlKC01MCUsIDUwJSk7XG59XG4ubm9VaS1ydGwgLndpZGdldC1zbGlkZXIgLm5vVWktdmFsdWUtaG9yaXpvbnRhbCxcbi5ub1VpLXJ0bCAuanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXZhbHVlLWhvcml6b250YWwge1xuICAtd2Via2l0LXRyYW5zZm9ybTogdHJhbnNsYXRlKDUwJSwgNTAlKTtcbiAgdHJhbnNmb3JtOiB0cmFuc2xhdGUoNTAlLCA1MCUpO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktbWFya2VyLWhvcml6b250YWwubm9VaS1tYXJrZXIsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLW1hcmtlci1ob3Jpem9udGFsLm5vVWktbWFya2VyIHtcbiAgbWFyZ2luLWxlZnQ6IC0xcHg7XG4gIHdpZHRoOiAycHg7XG4gIGhlaWdodDogNXB4O1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktbWFya2VyLWhvcml6b250YWwubm9VaS1tYXJrZXItc3ViLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1tYXJrZXItaG9yaXpvbnRhbC5ub1VpLW1hcmtlci1zdWIge1xuICBoZWlnaHQ6IDEwcHg7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS1tYXJrZXItaG9yaXpvbnRhbC5ub1VpLW1hcmtlci1sYXJnZSxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktbWFya2VyLWhvcml6b250YWwubm9VaS1tYXJrZXItbGFyZ2Uge1xuICBoZWlnaHQ6IDE1cHg7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS1waXBzLXZlcnRpY2FsLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1waXBzLXZlcnRpY2FsIHtcbiAgcGFkZGluZzogMCAxMHB4O1xuICBoZWlnaHQ6IDEwMCU7XG4gIHRvcDogMDtcbiAgbGVmdDogMTAwJTtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLXZhbHVlLXZlcnRpY2FsLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS12YWx1ZS12ZXJ0aWNhbCB7XG4gIC13ZWJraXQtdHJhbnNmb3JtOiB0cmFuc2xhdGUoMCwgLTUwJSk7XG4gIHRyYW5zZm9ybTogdHJhbnNsYXRlKDAsIC01MCUpO1xuICBwYWRkaW5nLWxlZnQ6IDI1cHg7XG59XG4ubm9VaS1ydGwgLndpZGdldC1zbGlkZXIgLm5vVWktdmFsdWUtdmVydGljYWwsXG4ubm9VaS1ydGwgLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS12YWx1ZS12ZXJ0aWNhbCB7XG4gIC13ZWJraXQtdHJhbnNmb3JtOiB0cmFuc2xhdGUoMCwgNTAlKTtcbiAgdHJhbnNmb3JtOiB0cmFuc2xhdGUoMCwgNTAlKTtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLW1hcmtlci12ZXJ0aWNhbC5ub1VpLW1hcmtlcixcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktbWFya2VyLXZlcnRpY2FsLm5vVWktbWFya2VyIHtcbiAgd2lkdGg6IDVweDtcbiAgaGVpZ2h0OiAycHg7XG4gIG1hcmdpbi10b3A6IC0xcHg7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS1tYXJrZXItdmVydGljYWwubm9VaS1tYXJrZXItc3ViLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1tYXJrZXItdmVydGljYWwubm9VaS1tYXJrZXItc3ViIHtcbiAgd2lkdGg6IDEwcHg7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS1tYXJrZXItdmVydGljYWwubm9VaS1tYXJrZXItbGFyZ2UsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLW1hcmtlci12ZXJ0aWNhbC5ub1VpLW1hcmtlci1sYXJnZSB7XG4gIHdpZHRoOiAxNXB4O1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktdG9vbHRpcCxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktdG9vbHRpcCB7XG4gIGRpc3BsYXk6IGJsb2NrO1xuICBwb3NpdGlvbjogYWJzb2x1dGU7XG4gIGJvcmRlcjogMXB4IHNvbGlkICNEOUQ5RDk7XG4gIGJvcmRlci1yYWRpdXM6IDNweDtcbiAgYmFja2dyb3VuZDogI2ZmZjtcbiAgY29sb3I6ICMwMDA7XG4gIHBhZGRpbmc6IDVweDtcbiAgdGV4dC1hbGlnbjogY2VudGVyO1xuICB3aGl0ZS1zcGFjZTogbm93cmFwO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktaG9yaXpvbnRhbCAubm9VaS10b29sdGlwLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1ob3Jpem9udGFsIC5ub1VpLXRvb2x0aXAge1xuICAtd2Via2l0LXRyYW5zZm9ybTogdHJhbnNsYXRlKC01MCUsIDApO1xuICB0cmFuc2Zvcm06IHRyYW5zbGF0ZSgtNTAlLCAwKTtcbiAgbGVmdDogNTAlO1xuICBib3R0b206IDEyMCU7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS12ZXJ0aWNhbCAubm9VaS10b29sdGlwLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS12ZXJ0aWNhbCAubm9VaS10b29sdGlwIHtcbiAgLXdlYmtpdC10cmFuc2Zvcm06IHRyYW5zbGF0ZSgwLCAtNTAlKTtcbiAgdHJhbnNmb3JtOiB0cmFuc2xhdGUoMCwgLTUwJSk7XG4gIHRvcDogNTAlO1xuICByaWdodDogMTIwJTtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLWhvcml6b250YWwgLm5vVWktb3JpZ2luID4gLm5vVWktdG9vbHRpcCxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktaG9yaXpvbnRhbCAubm9VaS1vcmlnaW4gPiAubm9VaS10b29sdGlwIHtcbiAgLXdlYmtpdC10cmFuc2Zvcm06IHRyYW5zbGF0ZSg1MCUsIDApO1xuICB0cmFuc2Zvcm06IHRyYW5zbGF0ZSg1MCUsIDApO1xuICBsZWZ0OiBhdXRvO1xuICBib3R0b206IDEwcHg7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS12ZXJ0aWNhbCAubm9VaS1vcmlnaW4gPiAubm9VaS10b29sdGlwLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS12ZXJ0aWNhbCAubm9VaS1vcmlnaW4gPiAubm9VaS10b29sdGlwIHtcbiAgLXdlYmtpdC10cmFuc2Zvcm06IHRyYW5zbGF0ZSgwLCAtMThweCk7XG4gIHRyYW5zZm9ybTogdHJhbnNsYXRlKDAsIC0xOHB4KTtcbiAgdG9wOiBhdXRvO1xuICByaWdodDogMjhweDtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLWNvbm5lY3QsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLWNvbm5lY3Qge1xuICBiYWNrZ3JvdW5kOiAjMjE5NmYzO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktaG9yaXpvbnRhbCxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktaG9yaXpvbnRhbCB7XG4gIGhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1zbGlkZXItdHJhY2stdGhpY2tuZXNzKTtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLXZlcnRpY2FsLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS12ZXJ0aWNhbCB7XG4gIHdpZHRoOiB2YXIoLS1qcC13aWRnZXRzLXNsaWRlci10cmFjay10aGlja25lc3MpO1xuICBoZWlnaHQ6IDEwMCU7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS1ob3Jpem9udGFsIC5ub1VpLWhhbmRsZSxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktaG9yaXpvbnRhbCAubm9VaS1oYW5kbGUge1xuICB3aWR0aDogdmFyKC0tanAtd2lkZ2V0cy1zbGlkZXItaGFuZGxlLXNpemUpO1xuICBoZWlnaHQ6IHZhcigtLWpwLXdpZGdldHMtc2xpZGVyLWhhbmRsZS1zaXplKTtcbiAgYm9yZGVyLXJhZGl1czogNTAlO1xuICB0b3A6IGNhbGMoKHZhcigtLWpwLXdpZGdldHMtc2xpZGVyLXRyYWNrLXRoaWNrbmVzcykgLSB2YXIoLS1qcC13aWRnZXRzLXNsaWRlci1oYW5kbGUtc2l6ZSkpIC8gMik7XG4gIHJpZ2h0OiBjYWxjKHZhcigtLWpwLXdpZGdldHMtc2xpZGVyLWhhbmRsZS1zaXplKSAvIC0yKTtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLXZlcnRpY2FsIC5ub1VpLWhhbmRsZSxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktdmVydGljYWwgLm5vVWktaGFuZGxlIHtcbiAgaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLXNsaWRlci1oYW5kbGUtc2l6ZSk7XG4gIHdpZHRoOiB2YXIoLS1qcC13aWRnZXRzLXNsaWRlci1oYW5kbGUtc2l6ZSk7XG4gIGJvcmRlci1yYWRpdXM6IDUwJTtcbiAgcmlnaHQ6IGNhbGMoKHZhcigtLWpwLXdpZGdldHMtc2xpZGVyLWhhbmRsZS1zaXplKSAtIHZhcigtLWpwLXdpZGdldHMtc2xpZGVyLXRyYWNrLXRoaWNrbmVzcykpIC8gLTIpO1xuICB0b3A6IGNhbGModmFyKC0tanAtd2lkZ2V0cy1zbGlkZXItaGFuZGxlLXNpemUpIC8gLTIpO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktaGFuZGxlOmFmdGVyLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1oYW5kbGU6YWZ0ZXIge1xuICBjb250ZW50OiBub25lO1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktaGFuZGxlOmJlZm9yZSxcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLm5vVWktaGFuZGxlOmJlZm9yZSB7XG4gIGNvbnRlbnQ6IG5vbmU7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS10YXJnZXQsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXRhcmdldCB7XG4gIGJhY2tncm91bmQ6ICNmYWZhZmE7XG4gIGJvcmRlci1yYWRpdXM6IDRweDtcbiAgYm9yZGVyOiAxcHg7XG4gIC8qIGJveC1zaGFkb3c6IGluc2V0IDAgMXB4IDFweCAjRjBGMEYwLCAwIDNweCA2cHggLTVweCAjQkJCOyAqL1xufVxuLndpZGdldC1zbGlkZXIgLnVpLXNsaWRlcixcbi5qdXB5dGVyLXdpZGdldC1zbGlkZXIgLnVpLXNsaWRlciB7XG4gIGJvcmRlcjogdmFyKC0tanAtd2lkZ2V0cy1zbGlkZXItYm9yZGVyLXdpZHRoKSBzb2xpZCB2YXIoLS1qcC1sYXlvdXQtY29sb3IzKTtcbiAgYmFja2dyb3VuZDogdmFyKC0tanAtbGF5b3V0LWNvbG9yMyk7XG4gIGJveC1zaXppbmc6IGJvcmRlci1ib3g7XG4gIHBvc2l0aW9uOiByZWxhdGl2ZTtcbiAgYm9yZGVyLXJhZGl1czogMHB4O1xufVxuLndpZGdldC1zbGlkZXIgLm5vVWktaGFuZGxlLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1oYW5kbGUge1xuICB3aWR0aDogdmFyKC0tanAtd2lkZ2V0cy1zbGlkZXItaGFuZGxlLXNpemUpO1xuICBib3JkZXI6IDFweCBzb2xpZCAjZDlkOWQ5O1xuICBib3JkZXItcmFkaXVzOiAzcHg7XG4gIGJhY2tncm91bmQ6ICNmZmY7XG4gIGN1cnNvcjogZGVmYXVsdDtcbiAgYm94LXNoYWRvdzogbm9uZTtcbiAgb3V0bGluZTogbm9uZTtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLXRhcmdldDpub3QoW2Rpc2FibGVkXSkgLm5vVWktaGFuZGxlOmhvdmVyLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS10YXJnZXQ6bm90KFtkaXNhYmxlZF0pIC5ub1VpLWhhbmRsZTpob3Zlcixcbi53aWRnZXQtc2xpZGVyIC5ub1VpLXRhcmdldDpub3QoW2Rpc2FibGVkXSkgLm5vVWktaGFuZGxlOmZvY3VzLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS10YXJnZXQ6bm90KFtkaXNhYmxlZF0pIC5ub1VpLWhhbmRsZTpmb2N1cyB7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLXdpZGdldHMtc2xpZGVyLWFjdGl2ZS1oYW5kbGUtY29sb3IpO1xuICBib3JkZXI6IHZhcigtLWpwLXdpZGdldHMtc2xpZGVyLWJvcmRlci13aWR0aCkgc29saWQgdmFyKC0tanAtd2lkZ2V0cy1zbGlkZXItYWN0aXZlLWhhbmRsZS1jb2xvcik7XG59XG4ud2lkZ2V0LXNsaWRlciBbZGlzYWJsZWRdLm5vVWktdGFyZ2V0LFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciBbZGlzYWJsZWRdLm5vVWktdGFyZ2V0IHtcbiAgb3BhY2l0eTogMC4zNTtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLWNvbm5lY3RzLFxuLmp1cHl0ZXItd2lkZ2V0LXNsaWRlciAubm9VaS1jb25uZWN0cyB7XG4gIG92ZXJmbG93OiB2aXNpYmxlO1xuICB6LWluZGV4OiAwO1xuICBiYWNrZ3JvdW5kOiB2YXIoLS1qcC1sYXlvdXQtY29sb3IzKTtcbn1cbi53aWRnZXQtc2xpZGVyIC5ub1VpLXZlcnRpY2FsIC5ub1VpLWNvbm5lY3QsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLXZlcnRpY2FsIC5ub1VpLWNvbm5lY3Qge1xuICB3aWR0aDogY2FsYygxMDAlICsgMnB4KTtcbiAgcmlnaHQ6IC0xcHg7XG59XG4ud2lkZ2V0LXNsaWRlciAubm9VaS1ob3Jpem9udGFsIC5ub1VpLWNvbm5lY3QsXG4uanVweXRlci13aWRnZXQtc2xpZGVyIC5ub1VpLWhvcml6b250YWwgLm5vVWktY29ubmVjdCB7XG4gIGhlaWdodDogY2FsYygxMDAlICsgMnB4KTtcbiAgdG9wOiAtMXB4O1xufVxuIl0sInNvdXJjZVJvb3QiOiIifQ== */</style><style>/* Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*
 * We assume that the CSS variables in
 * https://github.com/jupyterlab/jupyterlab/blob/master/src/default-theme/variables.css
 * have been defined.
 */

:root {
  --jp-widgets-color: var(--jp-content-font-color1);
  --jp-widgets-label-color: var(--jp-widgets-color);
  --jp-widgets-readout-color: var(--jp-widgets-color);
  --jp-widgets-font-size: var(--jp-ui-font-size1);
  --jp-widgets-margin: 2px;
  --jp-widgets-inline-height: 28px;
  --jp-widgets-inline-width: 300px;
  --jp-widgets-inline-width-short: calc(
    var(--jp-widgets-inline-width) / 2 - var(--jp-widgets-margin)
  );
  --jp-widgets-inline-width-tiny: calc(
    var(--jp-widgets-inline-width-short) / 2 - var(--jp-widgets-margin)
  );
  --jp-widgets-inline-margin: 4px; /* margin between inline elements */
  --jp-widgets-inline-label-width: 80px;
  --jp-widgets-border-width: var(--jp-border-width);
  --jp-widgets-vertical-height: 200px;
  --jp-widgets-horizontal-tab-height: 24px;
  --jp-widgets-horizontal-tab-width: 144px;
  --jp-widgets-horizontal-tab-top-border: 2px;
  --jp-widgets-progress-thickness: 20px;
  --jp-widgets-container-padding: 15px;
  --jp-widgets-input-padding: 4px;
  --jp-widgets-radio-item-height-adjustment: 8px;
  --jp-widgets-radio-item-height: calc(
    var(--jp-widgets-inline-height) -
      var(--jp-widgets-radio-item-height-adjustment)
  );
  --jp-widgets-slider-track-thickness: 4px;
  --jp-widgets-slider-border-width: var(--jp-widgets-border-width);
  --jp-widgets-slider-handle-size: 16px;
  --jp-widgets-slider-handle-border-color: var(--jp-border-color1);
  --jp-widgets-slider-handle-background-color: var(--jp-layout-color1);
  --jp-widgets-slider-active-handle-color: var(--jp-brand-color1);
  --jp-widgets-menu-item-height: 24px;
  --jp-widgets-dropdown-arrow: url(data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjIuMSwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAxOCAxOCIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgMTggMTg7IiB4bWw6c3BhY2U9InByZXNlcnZlIj4KPHN0eWxlIHR5cGU9InRleHQvY3NzIj4KCS5zdDB7ZmlsbDpub25lO30KPC9zdHlsZT4KPHBhdGggZD0iTTUuMiw1LjlMOSw5LjdsMy44LTMuOGwxLjIsMS4ybC00LjksNWwtNC45LTVMNS4yLDUuOXoiLz4KPHBhdGggY2xhc3M9InN0MCIgZD0iTTAtMC42aDE4djE4SDBWLTAuNnoiLz4KPC9zdmc+Cg);
  --jp-widgets-input-color: var(--jp-ui-font-color1);
  --jp-widgets-input-background-color: var(--jp-layout-color1);
  --jp-widgets-input-border-color: var(--jp-border-color1);
  --jp-widgets-input-focus-border-color: var(--jp-brand-color2);
  --jp-widgets-input-border-width: var(--jp-widgets-border-width);
  --jp-widgets-disabled-opacity: 0.6;

  /* From Material Design Lite */
  --md-shadow-key-umbra-opacity: 0.2;
  --md-shadow-key-penumbra-opacity: 0.14;
  --md-shadow-ambient-shadow-opacity: 0.12;
}

.jupyter-widgets {
  margin: var(--jp-widgets-margin);
  box-sizing: border-box;
  color: var(--jp-widgets-color);
  overflow: visible;
}

.jp-Output-result > .jupyter-widgets {
  margin-left: 0;
  margin-right: 0;
}

/* vbox and hbox */

/* <DEPRECATED> */
.widget-inline-hbox, /* </DEPRECATED> */
 .jupyter-widget-inline-hbox {
  /* Horizontal widgets */
  box-sizing: border-box;
  display: flex;
  flex-direction: row;
  align-items: baseline;
}

/* <DEPRECATED> */
.widget-inline-vbox, /* </DEPRECATED> */
 .jupyter-widget-inline-vbox {
  /* Vertical Widgets */
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  align-items: center;
}

/* <DEPRECATED> */
.widget-box, /* </DEPRECATED> */
.jupyter-widget-box {
  box-sizing: border-box;
  display: flex;
  margin: 0;
  overflow: auto;
}

/* <DEPRECATED> */
.widget-gridbox, /* </DEPRECATED> */
.jupyter-widget-gridbox {
  box-sizing: border-box;
  display: grid;
  margin: 0;
  overflow: auto;
}

/* <DEPRECATED> */
.widget-hbox, /* </DEPRECATED> */
.jupyter-widget-hbox {
  flex-direction: row;
}

/* <DEPRECATED> */
.widget-vbox, /* </DEPRECATED> */
.jupyter-widget-vbox {
  flex-direction: column;
}

/* General Tags Styling */

.jupyter-widget-tagsinput {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  align-items: center;
  overflow: auto;

  cursor: text;
}

.jupyter-widget-tag {
  padding-left: 10px;
  padding-right: 10px;
  padding-top: 0px;
  padding-bottom: 0px;
  display: inline-block;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  text-align: center;
  font-size: var(--jp-widgets-font-size);

  height: calc(var(--jp-widgets-inline-height) - 2px);
  border: 0px solid;
  line-height: calc(var(--jp-widgets-inline-height) - 2px);
  box-shadow: none;

  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color2);
  border-color: var(--jp-border-color2);
  border: none;
  user-select: none;

  cursor: grab;
  transition: margin-left 200ms;
  margin: 1px 1px 1px 1px;
}

.jupyter-widget-tag.mod-active {
  /* MD Lite 4dp shadow */
  box-shadow: 0 4px 5px 0 rgba(0, 0, 0, var(--md-shadow-key-penumbra-opacity)),
    0 1px 10px 0 rgba(0, 0, 0, var(--md-shadow-ambient-shadow-opacity)),
    0 2px 4px -1px rgba(0, 0, 0, var(--md-shadow-key-umbra-opacity));
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color3);
}

.jupyter-widget-colortag {
  color: var(--jp-inverse-ui-font-color1);
}

.jupyter-widget-colortag.mod-active {
  color: var(--jp-inverse-ui-font-color0);
}

.jupyter-widget-taginput {
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-layout-color0);

  cursor: text;
  text-align: left;
}

.jupyter-widget-taginput:focus {
  outline: none;
}

.jupyter-widget-tag-close {
  margin-left: var(--jp-widgets-inline-margin);
  padding: 2px 0px 2px 2px;
}

.jupyter-widget-tag-close:hover {
  cursor: pointer;
}

/* Tag "Primary" Styling */

.jupyter-widget-tag.mod-primary {
  color: var(--jp-inverse-ui-font-color1);
  background-color: var(--jp-brand-color1);
}

.jupyter-widget-tag.mod-primary.mod-active {
  color: var(--jp-inverse-ui-font-color0);
  background-color: var(--jp-brand-color0);
}

/* Tag "Success" Styling */

.jupyter-widget-tag.mod-success {
  color: var(--jp-inverse-ui-font-color1);
  background-color: var(--jp-success-color1);
}

.jupyter-widget-tag.mod-success.mod-active {
  color: var(--jp-inverse-ui-font-color0);
  background-color: var(--jp-success-color0);
}

/* Tag "Info" Styling */

.jupyter-widget-tag.mod-info {
  color: var(--jp-inverse-ui-font-color1);
  background-color: var(--jp-info-color1);
}

.jupyter-widget-tag.mod-info.mod-active {
  color: var(--jp-inverse-ui-font-color0);
  background-color: var(--jp-info-color0);
}

/* Tag "Warning" Styling */

.jupyter-widget-tag.mod-warning {
  color: var(--jp-inverse-ui-font-color1);
  background-color: var(--jp-warn-color1);
}

.jupyter-widget-tag.mod-warning.mod-active {
  color: var(--jp-inverse-ui-font-color0);
  background-color: var(--jp-warn-color0);
}

/* Tag "Danger" Styling */

.jupyter-widget-tag.mod-danger {
  color: var(--jp-inverse-ui-font-color1);
  background-color: var(--jp-error-color1);
}

.jupyter-widget-tag.mod-danger.mod-active {
  color: var(--jp-inverse-ui-font-color0);
  background-color: var(--jp-error-color0);
}

/* General Button Styling */

.jupyter-button {
  padding-left: 10px;
  padding-right: 10px;
  padding-top: 0px;
  padding-bottom: 0px;
  display: inline-block;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  text-align: center;
  font-size: var(--jp-widgets-font-size);
  cursor: pointer;

  height: var(--jp-widgets-inline-height);
  border: 0px solid;
  line-height: var(--jp-widgets-inline-height);
  box-shadow: none;

  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color2);
  border-color: var(--jp-border-color2);
  border: none;
  user-select: none;
}

.jupyter-button i.fa {
  margin-right: var(--jp-widgets-inline-margin);
  pointer-events: none;
}

.jupyter-button:empty:before {
  content: '\200b'; /* zero-width space */
}

.jupyter-widgets.jupyter-button:disabled {
  opacity: var(--jp-widgets-disabled-opacity);
}

.jupyter-button i.fa.center {
  margin-right: 0;
}

.jupyter-button:hover:enabled,
.jupyter-button:focus:enabled {
  /* MD Lite 2dp shadow */
  box-shadow: 0 2px 2px 0 rgba(0, 0, 0, var(--md-shadow-key-penumbra-opacity)),
    0 3px 1px -2px rgba(0, 0, 0, var(--md-shadow-key-umbra-opacity)),
    0 1px 5px 0 rgba(0, 0, 0, var(--md-shadow-ambient-shadow-opacity));
}

.jupyter-button:active,
.jupyter-button.mod-active {
  /* MD Lite 4dp shadow */
  box-shadow: 0 4px 5px 0 rgba(0, 0, 0, var(--md-shadow-key-penumbra-opacity)),
    0 1px 10px 0 rgba(0, 0, 0, var(--md-shadow-ambient-shadow-opacity)),
    0 2px 4px -1px rgba(0, 0, 0, var(--md-shadow-key-umbra-opacity));
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color3);
}

.jupyter-button:focus:enabled {
  outline: 1px solid var(--jp-widgets-input-focus-border-color);
}

/* Button "Primary" Styling */

.jupyter-button.mod-primary {
  color: var(--jp-ui-inverse-font-color1);
  background-color: var(--jp-brand-color1);
}

.jupyter-button.mod-primary.mod-active {
  color: var(--jp-ui-inverse-font-color0);
  background-color: var(--jp-brand-color0);
}

.jupyter-button.mod-primary:active {
  color: var(--jp-ui-inverse-font-color0);
  background-color: var(--jp-brand-color0);
}

/* Button "Success" Styling */

.jupyter-button.mod-success {
  color: var(--jp-ui-inverse-font-color1);
  background-color: var(--jp-success-color1);
}

.jupyter-button.mod-success.mod-active {
  color: var(--jp-ui-inverse-font-color0);
  background-color: var(--jp-success-color0);
}

.jupyter-button.mod-success:active {
  color: var(--jp-ui-inverse-font-color0);
  background-color: var(--jp-success-color0);
}

/* Button "Info" Styling */

.jupyter-button.mod-info {
  color: var(--jp-ui-inverse-font-color1);
  background-color: var(--jp-info-color1);
}

.jupyter-button.mod-info.mod-active {
  color: var(--jp-ui-inverse-font-color0);
  background-color: var(--jp-info-color0);
}

.jupyter-button.mod-info:active {
  color: var(--jp-ui-inverse-font-color0);
  background-color: var(--jp-info-color0);
}

/* Button "Warning" Styling */

.jupyter-button.mod-warning {
  color: var(--jp-ui-inverse-font-color1);
  background-color: var(--jp-warn-color1);
}

.jupyter-button.mod-warning.mod-active {
  color: var(--jp-ui-inverse-font-color0);
  background-color: var(--jp-warn-color0);
}

.jupyter-button.mod-warning:active {
  color: var(--jp-ui-inverse-font-color0);
  background-color: var(--jp-warn-color0);
}

/* Button "Danger" Styling */

.jupyter-button.mod-danger {
  color: var(--jp-ui-inverse-font-color1);
  background-color: var(--jp-error-color1);
}

.jupyter-button.mod-danger.mod-active {
  color: var(--jp-ui-inverse-font-color0);
  background-color: var(--jp-error-color0);
}

.jupyter-button.mod-danger:active {
  color: var(--jp-ui-inverse-font-color0);
  background-color: var(--jp-error-color0);
}

/* Widget Button, Widget Toggle Button, Widget Upload */

/* <DEPRECATED> */
.widget-button, /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-toggle-button, /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-upload, /* </DEPRECATED> */
.jupyter-widget-button,
.jupyter-widget-toggle-button,
.jupyter-widget-upload {
  width: var(--jp-widgets-inline-width-short);
}

/* Widget Label Styling */

/* Override Bootstrap label css */
.jupyter-widgets label {
  margin-bottom: initial;
}

/* <DEPRECATED> */
.widget-label-basic, /* </DEPRECATED> */
.jupyter-widget-label-basic {
  /* Basic Label */
  color: var(--jp-widgets-label-color);
  font-size: var(--jp-widgets-font-size);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  line-height: var(--jp-widgets-inline-height);
}

/* <DEPRECATED> */
.widget-label, /* </DEPRECATED> */
.jupyter-widget-label {
  /* Label */
  color: var(--jp-widgets-label-color);
  font-size: var(--jp-widgets-font-size);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  line-height: var(--jp-widgets-inline-height);
}

/* <DEPRECATED> */
.widget-inline-hbox .widget-label, /* </DEPRECATED> */
.jupyter-widget-inline-hbox .jupyter-widget-label {
  /* Horizontal Widget Label */
  color: var(--jp-widgets-label-color);
  text-align: right;
  margin-right: calc(var(--jp-widgets-inline-margin) * 2);
  width: var(--jp-widgets-inline-label-width);
  flex-shrink: 0;
}

/* <DEPRECATED> */
.widget-inline-vbox .widget-label, /* </DEPRECATED> */
.jupyter-widget-inline-vbox .jupyter-widget-label {
  /* Vertical Widget Label */
  color: var(--jp-widgets-label-color);
  text-align: center;
  line-height: var(--jp-widgets-inline-height);
}

/* Widget Readout Styling */

/* <DEPRECATED> */
.widget-readout, /* </DEPRECATED> */
.jupyter-widget-readout {
  color: var(--jp-widgets-readout-color);
  font-size: var(--jp-widgets-font-size);
  height: var(--jp-widgets-inline-height);
  line-height: var(--jp-widgets-inline-height);
  overflow: hidden;
  white-space: nowrap;
  text-align: center;
}

/* <DEPRECATED> */
.widget-readout.overflow, /* </DEPRECATED> */
.jupyter-widget-readout.overflow {
  /* Overflowing Readout */

  /* From Material Design Lite
        shadow-key-umbra-opacity: 0.2;
        shadow-key-penumbra-opacity: 0.14;
        shadow-ambient-shadow-opacity: 0.12;
     */
  -webkit-box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.2),
    0 3px 1px -2px rgba(0, 0, 0, 0.14), 0 1px 5px 0 rgba(0, 0, 0, 0.12);

  -moz-box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.2),
    0 3px 1px -2px rgba(0, 0, 0, 0.14), 0 1px 5px 0 rgba(0, 0, 0, 0.12);

  box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.2), 0 3px 1px -2px rgba(0, 0, 0, 0.14),
    0 1px 5px 0 rgba(0, 0, 0, 0.12);
}

/* <DEPRECATED> */
.widget-inline-hbox .widget-readout, /* </DEPRECATED> */
.jupyter-widget-inline-hbox .jupyter-widget-readout {
  /* Horizontal Readout */
  text-align: center;
  max-width: var(--jp-widgets-inline-width-short);
  min-width: var(--jp-widgets-inline-width-tiny);
  margin-left: var(--jp-widgets-inline-margin);
}

/* <DEPRECATED> */
.widget-inline-vbox .widget-readout, /* </DEPRECATED> */
.jupyter-widget-inline-vbox .jupyter-widget-readout {
  /* Vertical Readout */
  margin-top: var(--jp-widgets-inline-margin);
  /* as wide as the widget */
  width: inherit;
}

/* Widget Checkbox Styling */

/* <DEPRECATED> */
.widget-checkbox, /* </DEPRECATED> */
.jupyter-widget-checkbox {
  width: var(--jp-widgets-inline-width);
  height: var(--jp-widgets-inline-height);
  line-height: var(--jp-widgets-inline-height);
}

/* <DEPRECATED> */
.widget-checkbox input[type='checkbox'], /* </DEPRECATED> */
.jupyter-widget-checkbox input[type='checkbox'] {
  margin: 0px calc(var(--jp-widgets-inline-margin) * 2) 0px 0px;
  line-height: var(--jp-widgets-inline-height);
  font-size: large;
  flex-grow: 1;
  flex-shrink: 0;
  align-self: center;
}

/* Widget Valid Styling */

/* <DEPRECATED> */
.widget-valid, /* </DEPRECATED> */
.jupyter-widget-valid {
  height: var(--jp-widgets-inline-height);
  line-height: var(--jp-widgets-inline-height);
  width: var(--jp-widgets-inline-width-short);
  font-size: var(--jp-widgets-font-size);
}

/* <DEPRECATED> */
.widget-valid i, /* </DEPRECATED> */
.jupyter-widget-valid i {
  line-height: var(--jp-widgets-inline-height);
  margin-right: var(--jp-widgets-inline-margin);
  margin-left: var(--jp-widgets-inline-margin);
}

/* <DEPRECATED> */
.widget-valid.mod-valid i, /* </DEPRECATED> */
.jupyter-widget-valid.mod-valid i {
  color: green;
}

/* <DEPRECATED> */
.widget-valid.mod-invalid i, /* </DEPRECATED> */
.jupyter-widget-valid.mod-invalid i {
  color: red;
}

/* <DEPRECATED> */
.widget-valid.mod-valid .widget-valid-readout, /* </DEPRECATED> */
.jupyter-widget-valid.mod-valid .jupyter-widget-valid-readout {
  display: none;
}

/* Widget Text and TextArea Styling */

/* <DEPRECATED> */
.widget-textarea, /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-text, /* </DEPRECATED> */
.jupyter-widget-textarea,
.jupyter-widget-text {
  width: var(--jp-widgets-inline-width);
}

/* <DEPRECATED> */
.widget-text input[type='text'], /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-text input[type='number'], /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-text input[type='password'], /* </DEPRECATED> */
.jupyter-widget-text input[type='text'],
.jupyter-widget-text input[type='number'],
.jupyter-widget-text input[type='password'] {
  height: var(--jp-widgets-inline-height);
}

/* <DEPRECATED> */
.widget-text input[type='text']:disabled, /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-text input[type='number']:disabled, /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-text input[type='password']:disabled, /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-textarea textarea:disabled, /* </DEPRECATED> */
.jupyter-widget-text input[type='text']:disabled,
.jupyter-widget-text input[type='number']:disabled,
.jupyter-widget-text input[type='password']:disabled,
.jupyter-widget-textarea textarea:disabled {
  opacity: var(--jp-widgets-disabled-opacity);
}

/* <DEPRECATED> */
.widget-text input[type='text'], /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-text input[type='number'], /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-text input[type='password'], /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-textarea textarea, /* </DEPRECATED> */
.jupyter-widget-text input[type='text'],
.jupyter-widget-text input[type='number'],
.jupyter-widget-text input[type='password'],
.jupyter-widget-textarea textarea {
  box-sizing: border-box;
  border: var(--jp-widgets-input-border-width) solid
    var(--jp-widgets-input-border-color);
  background-color: var(--jp-widgets-input-background-color);
  color: var(--jp-widgets-input-color);
  font-size: var(--jp-widgets-font-size);
  flex-grow: 1;
  min-width: 0; /* This makes it possible for the flexbox to shrink this input */
  flex-shrink: 1;
  outline: none !important;
}

/* <DEPRECATED> */
.widget-text input[type='text'], /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-text input[type='password'], /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-textarea textarea, /* </DEPRECATED> */
.jupyter-widget-text input[type='text'],
.jupyter-widget-text input[type='password'],
.jupyter-widget-textarea textarea {
  padding: var(--jp-widgets-input-padding)
    calc(var(--jp-widgets-input-padding) * 2);
}

/* <DEPRECATED> */
.widget-text input[type='number'], /* </DEPRECATED> */
.jupyter-widget-text input[type='number'] {
  padding: var(--jp-widgets-input-padding) 0 var(--jp-widgets-input-padding)
    calc(var(--jp-widgets-input-padding) * 2);
}

/* <DEPRECATED> */
.widget-textarea textarea, /* </DEPRECATED> */
.jupyter-widget-textarea textarea {
  height: inherit;
  width: inherit;
}

/* <DEPRECATED> */
.widget-text input:focus, /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-textarea textarea:focus, /* </DEPRECATED> */
.jupyter-widget-text input:focus,
.jupyter-widget-textarea textarea:focus {
  border-color: var(--jp-widgets-input-focus-border-color);
}

/* Horizontal Slider */
/* <DEPRECATED> */
.widget-hslider, /* </DEPRECATED> */
.jupyter-widget-hslider {
  width: var(--jp-widgets-inline-width);
  height: var(--jp-widgets-inline-height);
  line-height: var(--jp-widgets-inline-height);

  /* Override the align-items baseline. This way, the description and readout
    still seem to align their baseline properly, and we don't have to have
    align-self: stretch in the .slider-container. */
  align-items: center;
}

/* <DEPRECATED> */
.widgets-slider .slider-container, /* </DEPRECATED> */
.jupyter-widgets-slider .slider-container {
  overflow: visible;
}

/* <DEPRECATED> */
.widget-hslider .slider-container, /* </DEPRECATED> */
.jupyter-widget-hslider .slider-container {
  margin-left: calc(
    var(--jp-widgets-slider-handle-size) / 2 - 2 *
      var(--jp-widgets-slider-border-width)
  );
  margin-right: calc(
    var(--jp-widgets-slider-handle-size) / 2 - 2 *
      var(--jp-widgets-slider-border-width)
  );
  flex: 1 1 var(--jp-widgets-inline-width-short);
}

/* Vertical Slider */

/* <DEPRECATED> */
.widget-vbox .widget-label, /* </DEPRECATED> */
.jupyter-widget-vbox .jupyter-widget-label {
  height: var(--jp-widgets-inline-height);
  line-height: var(--jp-widgets-inline-height);
}

/* <DEPRECATED> */
.widget-vslider, /* </DEPRECATED> */
.jupyter-widget-vslider {
  /* Vertical Slider */
  height: var(--jp-widgets-vertical-height);
  width: var(--jp-widgets-inline-width-tiny);
}

/* <DEPRECATED> */
.widget-vslider .slider-container, /* </DEPRECATED> */
.jupyter-widget-vslider .slider-container {
  flex: 1 1 var(--jp-widgets-inline-width-short);
  margin-left: auto;
  margin-right: auto;
  margin-bottom: calc(
    var(--jp-widgets-slider-handle-size) / 2 - 2 *
      var(--jp-widgets-slider-border-width)
  );
  margin-top: calc(
    var(--jp-widgets-slider-handle-size) / 2 - 2 *
      var(--jp-widgets-slider-border-width)
  );
  display: flex;
  flex-direction: column;
}

/* Widget Progress Styling */

.progress-bar {
  -webkit-transition: none;
  -moz-transition: none;
  -ms-transition: none;
  -o-transition: none;
  transition: none;
}

.progress-bar {
  height: var(--jp-widgets-inline-height);
}

.progress-bar {
  background-color: var(--jp-brand-color1);
}

.progress-bar-success {
  background-color: var(--jp-success-color1);
}

.progress-bar-info {
  background-color: var(--jp-info-color1);
}

.progress-bar-warning {
  background-color: var(--jp-warn-color1);
}

.progress-bar-danger {
  background-color: var(--jp-error-color1);
}

.progress {
  background-color: var(--jp-layout-color2);
  border: none;
  box-shadow: none;
}

/* Horisontal Progress */

/* <DEPRECATED> */
.widget-hprogress, /* </DEPRECATED> */
.jupyter-widget-hprogress {
  /* Progress Bar */
  height: var(--jp-widgets-inline-height);
  line-height: var(--jp-widgets-inline-height);
  width: var(--jp-widgets-inline-width);
  align-items: center;
}

/* <DEPRECATED> */
.widget-hprogress .progress, /* </DEPRECATED> */
.jupyter-widget-hprogress .progress {
  flex-grow: 1;
  margin-top: var(--jp-widgets-input-padding);
  margin-bottom: var(--jp-widgets-input-padding);
  align-self: stretch;
  /* Override bootstrap style */
  height: initial;
}

/* Vertical Progress */

/* <DEPRECATED> */
.widget-vprogress, /* </DEPRECATED> */
.jupyter-widget-vprogress {
  height: var(--jp-widgets-vertical-height);
  width: var(--jp-widgets-inline-width-tiny);
}

/* <DEPRECATED> */
.widget-vprogress .progress, /* </DEPRECATED> */
.jupyter-widget-vprogress .progress {
  flex-grow: 1;
  width: var(--jp-widgets-progress-thickness);
  margin-left: auto;
  margin-right: auto;
  margin-bottom: 0;
}

/* Select Widget Styling */

/* <DEPRECATED> */
.widget-dropdown, /* </DEPRECATED> */
.jupyter-widget-dropdown {
  height: var(--jp-widgets-inline-height);
  width: var(--jp-widgets-inline-width);
  line-height: var(--jp-widgets-inline-height);
}

/* <DEPRECATED> */
.widget-dropdown > select, /* </DEPRECATED> */
.jupyter-widget-dropdown > select {
  padding-right: 20px;
  border: var(--jp-widgets-input-border-width) solid
    var(--jp-widgets-input-border-color);
  border-radius: 0;
  height: inherit;
  flex: 1 1 var(--jp-widgets-inline-width-short);
  min-width: 0; /* This makes it possible for the flexbox to shrink this input */
  box-sizing: border-box;
  outline: none !important;
  box-shadow: none;
  background-color: var(--jp-widgets-input-background-color);
  color: var(--jp-widgets-input-color);
  font-size: var(--jp-widgets-font-size);
  vertical-align: top;
  padding-left: calc(var(--jp-widgets-input-padding) * 2);
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  background-repeat: no-repeat;
  background-size: 20px;
  background-position: right center;
  background-image: var(--jp-widgets-dropdown-arrow);
}
/* <DEPRECATED> */
.widget-dropdown > select:focus, /* </DEPRECATED> */
.jupyter-widget-dropdown > select:focus {
  border-color: var(--jp-widgets-input-focus-border-color);
}

/* <DEPRECATED> */
.widget-dropdown > select:disabled, /* </DEPRECATED> */
.jupyter-widget-dropdown > select:disabled {
  opacity: var(--jp-widgets-disabled-opacity);
}

/* To disable the dotted border in Firefox around select controls.
   See http://stackoverflow.com/a/18853002 */
/* <DEPRECATED> */
.widget-dropdown > select:-moz-focusring, /* </DEPRECATED> */
.jupyter-widget-dropdown > select:-moz-focusring {
  color: transparent;
  text-shadow: 0 0 0 #000;
}

/* Select and SelectMultiple */

/* <DEPRECATED> */
.widget-select, /* </DEPRECATED> */
.jupyter-widget-select {
  width: var(--jp-widgets-inline-width);
  line-height: var(--jp-widgets-inline-height);

  /* Because Firefox defines the baseline of a select as the bottom of the
    control, we align the entire control to the top and add padding to the
    select to get an approximate first line baseline alignment. */
  align-items: flex-start;
}

/* <DEPRECATED> */
.widget-select > select, /* </DEPRECATED> */
.jupyter-widget-select > select {
  border: var(--jp-widgets-input-border-width) solid
    var(--jp-widgets-input-border-color);
  background-color: var(--jp-widgets-input-background-color);
  color: var(--jp-widgets-input-color);
  font-size: var(--jp-widgets-font-size);
  flex: 1 1 var(--jp-widgets-inline-width-short);
  outline: none !important;
  overflow: auto;
  height: inherit;

  /* Because Firefox defines the baseline of a select as the bottom of the
    control, we align the entire control to the top and add padding to the
    select to get an approximate first line baseline alignment. */
  padding-top: 5px;
}

/* <DEPRECATED> */
.widget-select > select:focus, /* </DEPRECATED> */
.jupyter-widget-select > select:focus {
  border-color: var(--jp-widgets-input-focus-border-color);
}

.wiget-select > select > option,
.jupyter-wiget-select > select > option {
  padding-left: var(--jp-widgets-input-padding);
  line-height: var(--jp-widgets-inline-height);
  /* line-height doesn't work on some browsers for select options */
  padding-top: calc(
    var(--jp-widgets-inline-height) - var(--jp-widgets-font-size) / 2
  );
  padding-bottom: calc(
    var(--jp-widgets-inline-height) - var(--jp-widgets-font-size) / 2
  );
}

/* Toggle Buttons Styling */

/* <DEPRECATED> */
.widget-toggle-buttons, /* </DEPRECATED> */
.jupyter-widget-toggle-buttons {
  line-height: var(--jp-widgets-inline-height);
}

/* <DEPRECATED> */
.widget-toggle-buttons .widget-toggle-button, /* </DEPRECATED> */
.jupyter-widget-toggle-buttons .jupyter-widget-toggle-button {
  margin-left: var(--jp-widgets-margin);
  margin-right: var(--jp-widgets-margin);
}

/* <DEPRECATED> */
.widget-toggle-buttons .jupyter-button:disabled, /* </DEPRECATED> */
.jupyter-widget-toggle-buttons .jupyter-button:disabled {
  opacity: var(--jp-widgets-disabled-opacity);
}

/* Radio Buttons Styling */

/* <DEPRECATED> */
.widget-radio, /* </DEPRECATED> */
.jupyter-widget-radio {
  width: var(--jp-widgets-inline-width);
  line-height: var(--jp-widgets-inline-height);
}

/* <DEPRECATED> */
.widget-radio-box, /* </DEPRECATED> */
.jupyter-widget-radio-box {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  box-sizing: border-box;
  flex-grow: 1;
  margin-bottom: var(--jp-widgets-radio-item-height-adjustment);
}

/* <DEPRECATED> */
.widget-radio-box label, /* </DEPRECATED> */
.jupyter-widget-radio-box label {
  height: var(--jp-widgets-radio-item-height);
  line-height: var(--jp-widgets-radio-item-height);
  font-size: var(--jp-widgets-font-size);
}

/* <DEPRECATED> */
.widget-radio-box input, /* </DEPRECATED> */
.jupyter-widget-radio-box input {
  height: var(--jp-widgets-radio-item-height);
  line-height: var(--jp-widgets-radio-item-height);
  margin: 0 calc(var(--jp-widgets-input-padding) * 2) 0 1px;
  float: left;
}

/* Color Picker Styling */

/* <DEPRECATED> */
.widget-colorpicker, /* </DEPRECATED> */
.jupyter-widget-colorpicker {
  width: var(--jp-widgets-inline-width);
  height: var(--jp-widgets-inline-height);
  line-height: var(--jp-widgets-inline-height);
}

/* <DEPRECATED> */
.widget-colorpicker > .widget-colorpicker-input, /* </DEPRECATED> */
.jupyter-widget-colorpicker > .jupyter-widget-colorpicker-input {
  flex-grow: 1;
  flex-shrink: 1;
  min-width: var(--jp-widgets-inline-width-tiny);
}

/* <DEPRECATED> */
.widget-colorpicker input[type='color'], /* </DEPRECATED> */
.jupyter-widget-colorpicker input[type='color'] {
  width: var(--jp-widgets-inline-height);
  height: var(--jp-widgets-inline-height);
  padding: 0 2px; /* make the color square actually square on Chrome on OS X */
  background: var(--jp-widgets-input-background-color);
  color: var(--jp-widgets-input-color);
  border: var(--jp-widgets-input-border-width) solid
    var(--jp-widgets-input-border-color);
  border-left: none;
  flex-grow: 0;
  flex-shrink: 0;
  box-sizing: border-box;
  align-self: stretch;
  outline: none !important;
}

/* <DEPRECATED> */
.widget-colorpicker.concise input[type='color'], /* </DEPRECATED> */
.jupyter-widget-colorpicker.concise input[type='color'] {
  border-left: var(--jp-widgets-input-border-width) solid
    var(--jp-widgets-input-border-color);
}

/* <DEPRECATED> */
.widget-colorpicker input[type='color']:focus, /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-colorpicker input[type='text']:focus, /* </DEPRECATED> */
.jupyter-widget-colorpicker input[type='color']:focus,
.jupyter-widget-colorpicker input[type='text']:focus {
  border-color: var(--jp-widgets-input-focus-border-color);
}

/* <DEPRECATED> */
.widget-colorpicker input[type='text'], /* </DEPRECATED> */
.jupyter-widget-colorpicker input[type='text'] {
  flex-grow: 1;
  outline: none !important;
  height: var(--jp-widgets-inline-height);
  line-height: var(--jp-widgets-inline-height);
  background: var(--jp-widgets-input-background-color);
  color: var(--jp-widgets-input-color);
  border: var(--jp-widgets-input-border-width) solid
    var(--jp-widgets-input-border-color);
  font-size: var(--jp-widgets-font-size);
  padding: var(--jp-widgets-input-padding)
    calc(var(--jp-widgets-input-padding) * 2);
  min-width: 0; /* This makes it possible for the flexbox to shrink this input */
  flex-shrink: 1;
  box-sizing: border-box;
}

/* <DEPRECATED> */
.widget-colorpicker input[type='text']:disabled, /* </DEPRECATED> */
.jupyter-widget-colorpicker input[type='text']:disabled {
  opacity: var(--jp-widgets-disabled-opacity);
}

/* Date Picker Styling */

/* <DEPRECATED> */
.widget-datepicker, /* </DEPRECATED> */
.jupyter-widget-datepicker {
  width: var(--jp-widgets-inline-width);
  height: var(--jp-widgets-inline-height);
  line-height: var(--jp-widgets-inline-height);
}

/* <DEPRECATED> */
.widget-datepicker input[type='date'], /* </DEPRECATED> */
.jupyter-widget-datepicker input[type='date'] {
  flex-grow: 1;
  flex-shrink: 1;
  min-width: 0; /* This makes it possible for the flexbox to shrink this input */
  outline: none !important;
  height: var(--jp-widgets-inline-height);
  border: var(--jp-widgets-input-border-width) solid
    var(--jp-widgets-input-border-color);
  background-color: var(--jp-widgets-input-background-color);
  color: var(--jp-widgets-input-color);
  font-size: var(--jp-widgets-font-size);
  padding: var(--jp-widgets-input-padding)
    calc(var(--jp-widgets-input-padding) * 2);
  box-sizing: border-box;
}

/* <DEPRECATED> */
.widget-datepicker input[type='date']:focus, /* </DEPRECATED> */
.jupyter-widget-datepicker input[type='date']:focus {
  border-color: var(--jp-widgets-input-focus-border-color);
}

/* <DEPRECATED> */
.widget-datepicker input[type='date']:invalid, /* </DEPRECATED> */
.jupyter-widget-datepicker input[type='date']:invalid {
  border-color: var(--jp-warn-color1);
}

/* <DEPRECATED> */
.widget-datepicker input[type='date']:disabled, /* </DEPRECATED> */
.jupyter-widget-datepicker input[type='date']:disabled {
  opacity: var(--jp-widgets-disabled-opacity);
}

/* Play Widget */

/* <DEPRECATED> */
.widget-play, /* </DEPRECATED> */
.jupyter-widget-play {
  width: var(--jp-widgets-inline-width-short);
  display: flex;
  align-items: stretch;
}

/* <DEPRECATED> */
.widget-play .jupyter-button, /* </DEPRECATED> */
.jupyter-widget-play .jupyter-button {
  flex-grow: 1;
  height: auto;
}

/* <DEPRECATED> */
.widget-play .jupyter-button:disabled, /* </DEPRECATED> */
.jupyter-widget-play .jupyter-button:disabled {
  opacity: var(--jp-widgets-disabled-opacity);
}

/* Tab Widget */

/* <DEPRECATED> */
.jupyter-widgets.widget-tab, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab {
  display: flex;
  flex-direction: column;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar {
  /* Necessary so that a tab can be shifted down to overlay the border of the box below. */
  overflow-x: visible;
  overflow-y: visible;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar > .p-TabBar-content, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar > .p-TabBar-content, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar > .lm-TabBar-content {
  /* Make sure that the tab grows from bottom up */
  align-items: flex-end;
  min-width: 0;
  min-height: 0;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .widget-tab-contents, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .widget-tab-contents {
  width: 100%;
  box-sizing: border-box;
  margin: 0;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding: var(--jp-widgets-container-padding);
  flex-grow: 1;
  overflow: auto;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar {
  font: var(--jp-widgets-font-size) Helvetica, Arial, sans-serif;
  min-height: calc(
    var(--jp-widgets-horizontal-tab-height) + var(--jp-border-width)
  );
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar .p-TabBar-tab, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tab {
  flex: 0 1 var(--jp-widgets-horizontal-tab-width);
  min-width: 35px;
  min-height: calc(
    var(--jp-widgets-horizontal-tab-height) + var(--jp-border-width)
  );
  line-height: var(--jp-widgets-horizontal-tab-height);
  margin-left: calc(-1 * var(--jp-border-width));
  padding: 0px 10px;
  background: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
  border: var(--jp-border-width) solid var(--jp-border-color1);
  border-bottom: none;
  position: relative;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab.p-mod-current, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar .p-TabBar-tab.p-mod-current, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tab.lm-mod-current {
  color: var(--jp-ui-font-color0);
  /* We want the background to match the tab content background */
  background: var(--jp-layout-color1);
  min-height: calc(
    var(--jp-widgets-horizontal-tab-height) + 2 * var(--jp-border-width)
  );
  transform: translateY(var(--jp-border-width));
  overflow: visible;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab.p-mod-current:before, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar .p-TabBar-tab.p-mod-current:before, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tab.lm-mod-current:before {
  position: absolute;
  top: calc(-1 * var(--jp-border-width));
  left: calc(-1 * var(--jp-border-width));
  content: '';
  height: var(--jp-widgets-horizontal-tab-top-border);
  width: calc(100% + 2 * var(--jp-border-width));
  background: var(--jp-brand-color1);
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab:first-child, /* </DEPRECATED> */
/* <DEPRECATED> */.jupyter-widgets.jupyter-widget-tab > .p-TabBar .p-TabBar-tab:first-child, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tab:first-child {
  margin-left: 0;
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab
  > .p-TabBar
  .p-TabBar-tab:hover:not(.p-mod-current),
/* </DEPRECATED> */
/* <DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
  > .p-TabBar
  .p-TabBar-tab:hover:not(.p-mod-current),
/* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
  > .lm-TabBar
  .lm-TabBar-tab:hover:not(.lm-mod-current) {
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab
  > .p-TabBar
  .p-mod-closable
  > .p-TabBar-tabCloseIcon,
/* </DEPRECATED> */
/* <DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
> .p-TabBar
.p-mod-closable
> .p-TabBar-tabCloseIcon,
/* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
  > .lm-TabBar
  .lm-mod-closable
  > .lm-TabBar-tabCloseIcon {
  margin-left: 4px;
}

/* This font-awesome strategy may not work across FA4 and FA5, but we don't
actually support closable tabs, so it really doesn't matter */
/* <DEPRECATED> */
.jupyter-widgets.widget-tab
  > .p-TabBar
  .p-mod-closable
  > .p-TabBar-tabCloseIcon:before,
/* </DEPRECATED> */
/* <DEPRECATED> */
.jupyter-widgets.jupyter-widget-widget-tab
> .p-TabBar
.p-mod-closable
> .p-TabBar-tabCloseIcon:before,
/* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab
  > .lm-TabBar
  .lm-mod-closable
  > .lm-TabBar-tabCloseIcon:before {
  font-family: FontAwesome;
  content: '\f00d'; /* close */
}

/* <DEPRECATED> */
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabIcon, /* </DEPRECATED> */
/* <DEPRECATED> */ .jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabLabel, /* </DEPRECATED> */
/* <DEPRECATED> */ .jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabCloseIcon, /* </DEPRECATED> */
/* <DEPRECATED> */ .jupyter-widgets.jupyter-widget-tab > .p-TabBar .p-TabBar-tabIcon, /* </DEPRECATED> */
/* <DEPRECATED> */ .jupyter-widgets.jupyter-widget-tab > .p-TabBar .p-TabBar-tabLabel, /* </DEPRECATED> */
/* <DEPRECATED> */ .jupyter-widgets.jupyter-widget-tab > .p-TabBar .p-TabBar-tabCloseIcon, /* </DEPRECATED> */
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tabIcon,
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tabLabel,
.jupyter-widgets.jupyter-widget-tab > .lm-TabBar .lm-TabBar-tabCloseIcon {
  line-height: var(--jp-widgets-horizontal-tab-height);
}

/* Accordion Widget */

.jupyter-widget-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
}

.jupyter-widget-Collapse-header {
  padding: var(--jp-widgets-input-padding);
  cursor: pointer;
  color: var(--jp-ui-font-color2);
  background-color: var(--jp-layout-color2);
  border: var(--jp-widgets-border-width) solid var(--jp-border-color1);
  padding: calc(var(--jp-widgets-container-padding) * 2 / 3)
    var(--jp-widgets-container-padding);
  font-weight: bold;
}

.jupyter-widget-Collapse-header:hover {
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
}

.jupyter-widget-Collapse-open > .jupyter-widget-Collapse-header {
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color0);
  cursor: default;
  border-bottom: none;
}

.jupyter-widget-Collapse-contents {
  padding: var(--jp-widgets-container-padding);
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  border-left: var(--jp-widgets-border-width) solid var(--jp-border-color1);
  border-right: var(--jp-widgets-border-width) solid var(--jp-border-color1);
  border-bottom: var(--jp-widgets-border-width) solid var(--jp-border-color1);
  overflow: auto;
}

.jupyter-widget-Accordion {
  display: flex;
  flex-direction: column;
  align-items: stretch;
}

.jupyter-widget-Accordion .jupyter-widget-Collapse {
  margin-bottom: 0;
}

.jupyter-widget-Accordion .jupyter-widget-Collapse + .jupyter-widget-Collapse {
  margin-top: 4px;
}

/* HTML widget */

/* <DEPRECATED> */
.widget-html, /* </DEPRECATED> */
/* <DEPRECATED> */ .widget-htmlmath, /* </DEPRECATED> */
.jupyter-widget-html,
.jupyter-widget-htmlmath {
  font-size: var(--jp-widgets-font-size);
}

/* <DEPRECATED> */
.widget-html > .widget-html-content, /* </DEPRECATED> */
/* <DEPRECATED> */.widget-htmlmath > .widget-html-content, /* </DEPRECATED> */
.jupyter-widget-html > .jupyter-widget-html-content,
.jupyter-widget-htmlmath > .jupyter-widget-html-content {
  /* Fill out the area in the HTML widget */
  align-self: stretch;
  flex-grow: 1;
  flex-shrink: 1;
  /* Makes sure the baseline is still aligned with other elements */
  line-height: var(--jp-widgets-inline-height);
  /* Make it possible to have absolutely-positioned elements in the html */
  position: relative;
}

/* Image widget  */

/* <DEPRECATED> */
.widget-image, /* </DEPRECATED> */
.jupyter-widget-image {
  max-width: 100%;
  height: auto;
}

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL3BhY2thZ2VzL2NvbnRyb2xzL2Nzcy93aWRnZXRzLWJhc2UuY3NzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOztFQUVFOztBQUVGOzs7O0VBSUU7O0FBS0Y7RUFDRSxpREFBaUQ7RUFDakQsaURBQWlEO0VBQ2pELG1EQUFtRDtFQUNuRCwrQ0FBK0M7RUFDL0Msd0JBQXdCO0VBQ3hCLGdDQUFnQztFQUNoQyxnQ0FBZ0M7RUFDaEM7O0dBRUM7RUFDRDs7R0FFQztFQUNELCtCQUErQixFQUFFLG1DQUFtQztFQUNwRSxxQ0FBcUM7RUFDckMsaURBQWlEO0VBQ2pELG1DQUFtQztFQUNuQyx3Q0FBd0M7RUFDeEMsd0NBQXdDO0VBQ3hDLDJDQUEyQztFQUMzQyxxQ0FBcUM7RUFDckMsb0NBQW9DO0VBQ3BDLCtCQUErQjtFQUMvQiw4Q0FBOEM7RUFDOUM7OztHQUdDO0VBQ0Qsd0NBQXdDO0VBQ3hDLGdFQUFnRTtFQUNoRSxxQ0FBcUM7RUFDckMsZ0VBQWdFO0VBQ2hFLG9FQUFvRTtFQUNwRSwrREFBK0Q7RUFDL0QsbUNBQW1DO0VBQ25DLG9FQUE0dUI7RUFDNXVCLGtEQUFrRDtFQUNsRCw0REFBNEQ7RUFDNUQsd0RBQXdEO0VBQ3hELDZEQUE2RDtFQUM3RCwrREFBK0Q7RUFDL0Qsa0NBQWtDOztFQUVsQyw4QkFBOEI7RUFDOUIsa0NBQWtDO0VBQ2xDLHNDQUFzQztFQUN0Qyx3Q0FBd0M7QUFDMUM7O0FBRUE7RUFDRSxnQ0FBZ0M7RUFDaEMsc0JBQXNCO0VBQ3RCLDhCQUE4QjtFQUM5QixpQkFBaUI7QUFDbkI7O0FBRUE7RUFDRSxjQUFjO0VBQ2QsZUFBZTtBQUNqQjs7QUFFQSxrQkFBa0I7O0FBRWxCLGlCQUFpQjtBQUNqQjs7RUFFRSx1QkFBdUI7RUFDdkIsc0JBQXNCO0VBQ3RCLGFBQWE7RUFDYixtQkFBbUI7RUFDbkIscUJBQXFCO0FBQ3ZCOztBQUVBLGlCQUFpQjtBQUNqQjs7RUFFRSxxQkFBcUI7RUFDckIsc0JBQXNC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QUNxQztFQUNyQyxpQkFBaUI7QUFDbkI7O0FBRUE7RUFDRSx5Q0FBeUM7RUFDekMsK0JBQStCO0FBQ2pDOztBQUVBO0VBQ0UseUNBQXlDO0VBQ3pDLCtCQUErQjtFQUMvQixlQUFlO0VBQ2YsbUJBQW1CO0FBQ3JCOztBQUVBO0VBQ0UsNENBQTRDO0VBQzVDLHlDQUF5QztFQUN6QywrQkFBK0I7RUFDL0IseUVBQXlFO0VBQ3pFLDBFQUEwRTtFQUMxRSwyRUFBMkU7RUFDM0UsY0FBYztBQUNoQjs7QUFFQTtFQUNFLGFBQWE7RUFDYixzQkFBc0I7RUFDdEIsb0JBQW9CO0FBQ3RCOztBQUVBO0VBQ0UsZ0JBQWdCO0FBQ2xCOztBQUVBO0VBQ0UsZUFBZTtBQUNqQjs7QUFFQSxnQkFBZ0I7O0FBRWhCLGlCQUFpQjtBQUNqQjs7OztFQUlFLHNDQUFzQztBQUN4Qzs7QUFFQSxpQkFBaUI7QUFDakI7Ozs7RUFJRSx5Q0FBeUM7RUFDekMsbUJBQW1CO0VBQ25CLFlBQVk7RUFDWixjQUFjO0VBQ2QsaUVBQWlFO0VBQ2pFLDRDQUE0QztFQUM1Qyx3RUFBd0U7RUFDeEUsa0JBQWtCO0FBQ3BCOztBQUVBLGtCQUFrQjs7QUFFbEIsaUJBQWlCO0FBQ2pCOztFQUVFLGVBQWU7RUFDZixZQUFZO0FBQ2QiLCJzb3VyY2VzQ29udGVudCI6WyIvKiBDb3B5cmlnaHQgKGMpIEp1cHl0ZXIgRGV2ZWxvcG1lbnQgVGVhbS5cbiAqIERpc3RyaWJ1dGVkIHVuZGVyIHRoZSB0ZXJtcyBvZiB0aGUgTW9kaWZpZWQgQlNEIExpY2Vuc2UuXG4gKi9cblxuLypcbiAqIFdlIGFzc3VtZSB0aGF0IHRoZSBDU1MgdmFyaWFibGVzIGluXG4gKiBodHRwczovL2dpdGh1Yi5jb20vanVweXRlcmxhYi9qdXB5dGVybGFiL2Jsb2IvbWFzdGVyL3NyYy9kZWZhdWx0LXRoZW1lL3ZhcmlhYmxlcy5jc3NcbiAqIGhhdmUgYmVlbiBkZWZpbmVkLlxuICovXG5cbkBpbXBvcnQgJy4vbHVtaW5vLmNzcyc7XG5AaW1wb3J0ICcuL25vdWlzbGlkZXIuY3NzJztcblxuOnJvb3Qge1xuICAtLWpwLXdpZGdldHMtY29sb3I6IHZhcigtLWpwLWNvbnRlbnQtZm9udC1jb2xvcjEpO1xuICAtLWpwLXdpZGdldHMtbGFiZWwtY29sb3I6IHZhcigtLWpwLXdpZGdldHMtY29sb3IpO1xuICAtLWpwLXdpZGdldHMtcmVhZG91dC1jb2xvcjogdmFyKC0tanAtd2lkZ2V0cy1jb2xvcik7XG4gIC0tanAtd2lkZ2V0cy1mb250LXNpemU6IHZhcigtLWpwLXVpLWZvbnQtc2l6ZTEpO1xuICAtLWpwLXdpZGdldHMtbWFyZ2luOiAycHg7XG4gIC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0OiAyOHB4O1xuICAtLWpwLXdpZGdldHMtaW5saW5lLXdpZHRoOiAzMDBweDtcbiAgLS1qcC13aWRnZXRzLWlubGluZS13aWR0aC1zaG9ydDogY2FsYyhcbiAgICB2YXIoLS1qcC13aWRnZXRzLWlubGluZS13aWR0aCkgLyAyIC0gdmFyKC0tanAtd2lkZ2V0cy1tYXJnaW4pXG4gICk7XG4gIC0tanAtd2lkZ2V0cy1pbmxpbmUtd2lkdGgtdGlueTogY2FsYyhcbiAgICB2YXIoLS1qcC13aWRnZXRzLWlubGluZS13aWR0aC1zaG9ydCkgLyAyIC0gdmFyKC0tanAtd2lkZ2V0cy1tYXJnaW4pXG4gICk7XG4gIC0tanAtd2lkZ2V0cy1pbmxpbmUtbWFyZ2luOiA0cHg7IC8qIG1hcmdpbiBiZXR3ZWVuIGlubGluZSBlbGVtZW50cyAqL1xuICAtLWpwLXdpZGdldHMtaW5saW5lLWxhYmVsLXdpZHRoOiA4MHB4O1xuICAtLWpwLXdpZGdldHMtYm9yZGVyLXdpZHRoOiB2YXIoLS1qcC1ib3JkZXItd2lkdGgpO1xuICAtLWpwLXdpZGdldHMtdmVydGljYWwtaGVpZ2h0OiAyMDBweDtcbiAgLS1qcC13aWRnZXRzLWhvcml6b250YWwtdGFiLWhlaWdodDogMjRweDtcbiAgLS1qcC13aWRnZXRzLWhvcml6b250YWwtdGFiLXdpZHRoOiAxNDRweDtcbiAgLS1qcC13aWRnZXRzLWhvcml6b250YWwtdGFiLXRvcC1ib3JkZXI6IDJweDtcbiAgLS1qcC13aWRnZXRzLXByb2dyZXNzLXRoaWNrbmVzczogMjBweDtcbiAgLS1qcC13aWRnZXRzLWNvbnRhaW5lci1wYWRkaW5nOiAxNXB4O1xuICAtLWpwLXdpZGdldHMtaW5wdXQtcGFkZGluZzogNHB4O1xuICAtLWpwLXdpZGdldHMtcmFkaW8taXRlbS1oZWlnaHQtYWRqdXN0bWVudDogOHB4O1xuICAtLWpwLXdpZGdldHMtcmFkaW8taXRlbS1oZWlnaHQ6IGNhbGMoXG4gICAgdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KSAtXG4gICAgICB2YXIoLS1qcC13aWRnZXRzLXJhZGlvLWl0ZW0taGVpZ2h0LWFkanVzdG1lbnQpXG4gICk7XG4gIC0tanAtd2lkZ2V0cy1zbGlkZXItdHJhY2stdGhpY2tuZXNzOiA0cHg7XG4gIC0tanAtd2lkZ2V0cy1zbGlkZXItYm9yZGVyLXdpZHRoOiB2YXIoLS1qcC13aWRnZXRzLWJvcmRlci13aWR0aCk7XG4gIC0tanAtd2lkZ2V0cy1zbGlkZXItaGFuZGxlLXNpemU6IDE2cHg7XG4gIC0tanAtd2lkZ2V0cy1zbGlkZXItaGFuZGxlLWJvcmRlci1jb2xvcjogdmFyKC0tanAtYm9yZGVyLWNvbG9yMSk7XG4gIC0tanAtd2lkZ2V0cy1zbGlkZXItaGFuZGxlLWJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLWxheW91dC1jb2xvcjEpO1xuICAtLWpwLXdpZGdldHMtc2xpZGVyLWFjdGl2ZS1oYW5kbGUtY29sb3I6IHZhcigtLWpwLWJyYW5kLWNvbG9yMSk7XG4gIC0tanAtd2lkZ2V0cy1tZW51LWl0ZW0taGVpZ2h0OiAyNHB4O1xuICAtLWpwLXdpZGdldHMtZHJvcGRvd24tYXJyb3c6IHVybCgnZGF0YTppbWFnZS9zdmcreG1sO2Jhc2U2NCxQRDk0Yld3Z2RtVnljMmx2YmowaU1TNHdJaUJsYm1OdlpHbHVaejBpZFhSbUxUZ2lQejRLUENFdExTQkhaVzVsY21GMGIzSTZJRUZrYjJKbElFbHNiSFZ6ZEhKaGRHOXlJREU1TGpJdU1Td2dVMVpISUVWNGNHOXlkQ0JRYkhWbkxVbHVJQzRnVTFaSElGWmxjbk5wYjI0NklEWXVNREFnUW5WcGJHUWdNQ2tnSUMwdFBnbzhjM1puSUhabGNuTnBiMjQ5SWpFdU1TSWdhV1E5SWt4aGVXVnlYekVpSUhodGJHNXpQU0pvZEhSd09pOHZkM2QzTG5jekxtOXlaeTh5TURBd0wzTjJaeUlnZUcxc2JuTTZlR3hwYm1zOUltaDBkSEE2THk5M2QzY3Vkek11YjNKbkx6RTVPVGt2ZUd4cGJtc2lJSGc5SWpCd2VDSWdlVDBpTUhCNElnb0pJSFpwWlhkQ2IzZzlJakFnTUNBeE9DQXhPQ0lnYzNSNWJHVTlJbVZ1WVdKc1pTMWlZV05yWjNKdmRXNWtPbTVsZHlBd0lEQWdNVGdnTVRnN0lpQjRiV3c2YzNCaFkyVTlJbkJ5WlhObGNuWmxJajRLUEhOMGVXeGxJSFI1Y0dVOUluUmxlSFF2WTNOeklqNEtDUzV6ZERCN1ptbHNiRHB1YjI1bE8zMEtQQzl6ZEhsc1pUNEtQSEJoZEdnZ1pEMGlUVFV1TWl3MUxqbE1PU3c1TGpkc015NDRMVE11T0d3eExqSXNNUzR5YkMwMExqa3NOV3d0TkM0NUxUVk1OUzR5TERVdU9Yb2lMejRLUEhCaGRHZ2dZMnhoYzNNOUluTjBNQ0lnWkQwaVRUQXRNQzQyYURFNGRqRTRTREJXTFRBdU5ub2lMejRLUEM5emRtYytDZycpO1xuICAtLWpwLXdpZGdldHMtaW5wdXQtY29sb3I6IHZhcigtLWpwLXVpLWZvbnQtY29sb3IxKTtcbiAgLS1qcC13aWRnZXRzLWlucHV0LWJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLWxheW91dC1jb2xvcjEpO1xuICAtLWpwLXdpZGdldHMtaW5wdXQtYm9yZGVyLWNvbG9yOiB2YXIoLS1qcC1ib3JkZXItY29sb3IxKTtcbiAgLS1qcC13aWRnZXRzLWlucHV0LWZvY3VzLWJvcmRlci1jb2xvcjogdmFyKC0tanAtYnJhbmQtY29sb3IyKTtcbiAgLS1qcC13aWRnZXRzLWlucHV0LWJvcmRlci13aWR0aDogdmFyKC0tanAtd2lkZ2V0cy1ib3JkZXItd2lkdGgpO1xuICAtLWpwLXdpZGdldHMtZGlzYWJsZWQtb3BhY2l0eTogMC42O1xuXG4gIC8qIEZyb20gTWF0ZXJpYWwgRGVzaWduIExpdGUgKi9cbiAgLS1tZC1zaGFkb3cta2V5LXVtYnJhLW9wYWNpdHk6IDAuMjtcbiAgLS1tZC1zaGFkb3cta2V5LXBlbnVtYnJhLW9wYWNpdHk6IDAuMTQ7XG4gIC0tbWQtc2hhZG93LWFtYmllbnQtc2hhZG93LW9wYWNpdHk6IDAuMTI7XG59XG5cbi5qdXB5dGVyLXdpZGdldHMge1xuICBtYXJnaW46IHZhcigtLWpwLXdpZGdldHMtbWFyZ2luKTtcbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcbiAgY29sb3I6IHZhcigtLWpwLXdpZGdldHMtY29sb3IpO1xuICBvdmVyZmxvdzogdmlzaWJsZTtcbn1cblxuLmpwLU91dHB1dC1yZXN1bHQgPiAuanVweXRlci13aWRnZXRzIHtcbiAgbWFyZ2luLWxlZnQ6IDA7XG4gIG1hcmdpbi1yaWdodDogMDtcbn1cblxuLyogdmJveCBhbmQgaGJveCAqL1xuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtaW5saW5lLWhib3gsIC8qIDwvREVQUkVDQVRFRD4gKi9cbiAuanVweXRlci13aWRnZXQtaW5saW5lLWhib3gge1xuICAvKiBIb3Jpem9udGFsIHdpZGdldHMgKi9cbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcbiAgZGlzcGxheTogZmxleDtcbiAgZmxleC1kaXJlY3Rpb246IHJvdztcbiAgYWxpZ24taXRlbXM6IGJhc2VsaW5lO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtaW5saW5lLXZib3gsIC8qIDwvREVQUkVDQVRFRD4gKi9cbiAuanVweXRlci13aWRnZXQtaW5saW5lLXZib3gge1xuICAvKiBWZXJ0aWNhbCBXaWRnZXRzICovXG4gIGJveC1zaXppbmc6IGJvcmRlci1ib3g7XG4gIGRpc3BsYXk6IGZsZXg7XG4gIGZsZXgtZGlyZWN0aW9uOiBjb2x1bW47XG4gIGFsaWduLWl0ZW1zOiBjZW50ZXI7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1ib3gsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1ib3gge1xuICBib3gtc2l6aW5nOiBib3JkZXItYm94O1xuICBkaXNwbGF5OiBmbGV4O1xuICBtYXJnaW46IDA7XG4gIG92ZXJmbG93OiBhdXRvO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtZ3JpZGJveCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWdyaWRib3gge1xuICBib3gtc2l6aW5nOiBib3JkZXItYm94O1xuICBkaXNwbGF5OiBncmlkO1xuICBtYXJnaW46IDA7XG4gIG92ZXJmbG93OiBhdXRvO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtaGJveCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWhib3gge1xuICBmbGV4LWRpcmVjdGlvbjogcm93O1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtdmJveCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LXZib3gge1xuICBmbGV4LWRpcmVjdGlvbjogY29sdW1uO1xufVxuXG4vKiBHZW5lcmFsIFRhZ3MgU3R5bGluZyAqL1xuXG4uanVweXRlci13aWRnZXQtdGFnc2lucHV0IHtcbiAgZGlzcGxheTogZmxleDtcbiAgZmxleC1kaXJlY3Rpb246IHJvdztcbiAgZmxleC13cmFwOiB3cmFwO1xuICBhbGlnbi1pdGVtczogY2VudGVyO1xuICBvdmVyZmxvdzogYXV0bztcblxuICBjdXJzb3I6IHRleHQ7XG59XG5cbi5qdXB5dGVyLXdpZGdldC10YWcge1xuICBwYWRkaW5nLWxlZnQ6IDEwcHg7XG4gIHBhZGRpbmctcmlnaHQ6IDEwcHg7XG4gIHBhZGRpbmctdG9wOiAwcHg7XG4gIHBhZGRpbmctYm90dG9tOiAwcHg7XG4gIGRpc3BsYXk6IGlubGluZS1ibG9jaztcbiAgd2hpdGUtc3BhY2U6IG5vd3JhcDtcbiAgb3ZlcmZsb3c6IGhpZGRlbjtcbiAgdGV4dC1vdmVyZmxvdzogZWxsaXBzaXM7XG4gIHRleHQtYWxpZ246IGNlbnRlcjtcbiAgZm9udC1zaXplOiB2YXIoLS1qcC13aWRnZXRzLWZvbnQtc2l6ZSk7XG5cbiAgaGVpZ2h0OiBjYWxjKHZhcigtLWpwLXdpZGdldHMtaW5saW5lLWhlaWdodCkgLSAycHgpO1xuICBib3JkZXI6IDBweCBzb2xpZDtcbiAgbGluZS1oZWlnaHQ6IGNhbGModmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KSAtIDJweCk7XG4gIGJveC1zaGFkb3c6IG5vbmU7XG5cbiAgY29sb3I6IHZhcigtLWpwLXVpLWZvbnQtY29sb3IxKTtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtbGF5b3V0LWNvbG9yMik7XG4gIGJvcmRlci1jb2xvcjogdmFyKC0tanAtYm9yZGVyLWNvbG9yMik7XG4gIGJvcmRlcjogbm9uZTtcbiAgdXNlci1zZWxlY3Q6IG5vbmU7XG5cbiAgY3Vyc29yOiBncmFiO1xuICB0cmFuc2l0aW9uOiBtYXJnaW4tbGVmdCAyMDBtcztcbiAgbWFyZ2luOiAxcHggMXB4IDFweCAxcHg7XG59XG5cbi5qdXB5dGVyLXdpZGdldC10YWcubW9kLWFjdGl2ZSB7XG4gIC8qIE1EIExpdGUgNGRwIHNoYWRvdyAqL1xuICBib3gtc2hhZG93OiAwIDRweCA1cHggMCByZ2JhKDAsIDAsIDAsIHZhcigtLW1kLXNoYWRvdy1rZXktcGVudW1icmEtb3BhY2l0eSkpLFxuICAgIDAgMXB4IDEwcHggMCByZ2JhKDAsIDAsIDAsIHZhcigtLW1kLXNoYWRvdy1hbWJpZW50LXNoYWRvdy1vcGFjaXR5KSksXG4gICAgMCAycHggNHB4IC0xcHggcmdiYSgwLCAwLCAwLCB2YXIoLS1tZC1zaGFkb3cta2V5LXVtYnJhLW9wYWNpdHkpKTtcbiAgY29sb3I6IHZhcigtLWpwLXVpLWZvbnQtY29sb3IxKTtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtbGF5b3V0LWNvbG9yMyk7XG59XG5cbi5qdXB5dGVyLXdpZGdldC1jb2xvcnRhZyB7XG4gIGNvbG9yOiB2YXIoLS1qcC1pbnZlcnNlLXVpLWZvbnQtY29sb3IxKTtcbn1cblxuLmp1cHl0ZXItd2lkZ2V0LWNvbG9ydGFnLm1vZC1hY3RpdmUge1xuICBjb2xvcjogdmFyKC0tanAtaW52ZXJzZS11aS1mb250LWNvbG9yMCk7XG59XG5cbi5qdXB5dGVyLXdpZGdldC10YWdpbnB1dCB7XG4gIGNvbG9yOiB2YXIoLS1qcC11aS1mb250LWNvbG9yMCk7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLWxheW91dC1jb2xvcjApO1xuXG4gIGN1cnNvcjogdGV4dDtcbiAgdGV4dC1hbGlnbjogbGVmdDtcbn1cblxuLmp1cHl0ZXItd2lkZ2V0LXRhZ2lucHV0OmZvY3VzIHtcbiAgb3V0bGluZTogbm9uZTtcbn1cblxuLmp1cHl0ZXItd2lkZ2V0LXRhZy1jbG9zZSB7XG4gIG1hcmdpbi1sZWZ0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1tYXJnaW4pO1xuICBwYWRkaW5nOiAycHggMHB4IDJweCAycHg7XG59XG5cbi5qdXB5dGVyLXdpZGdldC10YWctY2xvc2U6aG92ZXIge1xuICBjdXJzb3I6IHBvaW50ZXI7XG59XG5cbi8qIFRhZyBcIlByaW1hcnlcIiBTdHlsaW5nICovXG5cbi5qdXB5dGVyLXdpZGdldC10YWcubW9kLXByaW1hcnkge1xuICBjb2xvcjogdmFyKC0tanAtaW52ZXJzZS11aS1mb250LWNvbG9yMSk7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLWJyYW5kLWNvbG9yMSk7XG59XG5cbi5qdXB5dGVyLXdpZGdldC10YWcubW9kLXByaW1hcnkubW9kLWFjdGl2ZSB7XG4gIGNvbG9yOiB2YXIoLS1qcC1pbnZlcnNlLXVpLWZvbnQtY29sb3IwKTtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtYnJhbmQtY29sb3IwKTtcbn1cblxuLyogVGFnIFwiU3VjY2Vzc1wiIFN0eWxpbmcgKi9cblxuLmp1cHl0ZXItd2lkZ2V0LXRhZy5tb2Qtc3VjY2VzcyB7XG4gIGNvbG9yOiB2YXIoLS1qcC1pbnZlcnNlLXVpLWZvbnQtY29sb3IxKTtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtc3VjY2Vzcy1jb2xvcjEpO1xufVxuXG4uanVweXRlci13aWRnZXQtdGFnLm1vZC1zdWNjZXNzLm1vZC1hY3RpdmUge1xuICBjb2xvcjogdmFyKC0tanAtaW52ZXJzZS11aS1mb250LWNvbG9yMCk7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLXN1Y2Nlc3MtY29sb3IwKTtcbn1cblxuLyogVGFnIFwiSW5mb1wiIFN0eWxpbmcgKi9cblxuLmp1cHl0ZXItd2lkZ2V0LXRhZy5tb2QtaW5mbyB7XG4gIGNvbG9yOiB2YXIoLS1qcC1pbnZlcnNlLXVpLWZvbnQtY29sb3IxKTtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtaW5mby1jb2xvcjEpO1xufVxuXG4uanVweXRlci13aWRnZXQtdGFnLm1vZC1pbmZvLm1vZC1hY3RpdmUge1xuICBjb2xvcjogdmFyKC0tanAtaW52ZXJzZS11aS1mb250LWNvbG9yMCk7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLWluZm8tY29sb3IwKTtcbn1cblxuLyogVGFnIFwiV2FybmluZ1wiIFN0eWxpbmcgKi9cblxuLmp1cHl0ZXItd2lkZ2V0LXRhZy5tb2Qtd2FybmluZyB7XG4gIGNvbG9yOiB2YXIoLS1qcC1pbnZlcnNlLXVpLWZvbnQtY29sb3IxKTtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtd2Fybi1jb2xvcjEpO1xufVxuXG4uanVweXRlci13aWRnZXQtdGFnLm1vZC13YXJuaW5nLm1vZC1hY3RpdmUge1xuICBjb2xvcjogdmFyKC0tanAtaW52ZXJzZS11aS1mb250LWNvbG9yMCk7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLXdhcm4tY29sb3IwKTtcbn1cblxuLyogVGFnIFwiRGFuZ2VyXCIgU3R5bGluZyAqL1xuXG4uanVweXRlci13aWRnZXQtdGFnLm1vZC1kYW5nZXIge1xuICBjb2xvcjogdmFyKC0tanAtaW52ZXJzZS11aS1mb250LWNvbG9yMSk7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLWVycm9yLWNvbG9yMSk7XG59XG5cbi5qdXB5dGVyLXdpZGdldC10YWcubW9kLWRhbmdlci5tb2QtYWN0aXZlIHtcbiAgY29sb3I6IHZhcigtLWpwLWludmVyc2UtdWktZm9udC1jb2xvcjApO1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC1lcnJvci1jb2xvcjApO1xufVxuXG4vKiBHZW5lcmFsIEJ1dHRvbiBTdHlsaW5nICovXG5cbi5qdXB5dGVyLWJ1dHRvbiB7XG4gIHBhZGRpbmctbGVmdDogMTBweDtcbiAgcGFkZGluZy1yaWdodDogMTBweDtcbiAgcGFkZGluZy10b3A6IDBweDtcbiAgcGFkZGluZy1ib3R0b206IDBweDtcbiAgZGlzcGxheTogaW5saW5lLWJsb2NrO1xuICB3aGl0ZS1zcGFjZTogbm93cmFwO1xuICBvdmVyZmxvdzogaGlkZGVuO1xuICB0ZXh0LW92ZXJmbG93OiBlbGxpcHNpcztcbiAgdGV4dC1hbGlnbjogY2VudGVyO1xuICBmb250LXNpemU6IHZhcigtLWpwLXdpZGdldHMtZm9udC1zaXplKTtcbiAgY3Vyc29yOiBwb2ludGVyO1xuXG4gIGhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KTtcbiAgYm9yZGVyOiAwcHggc29saWQ7XG4gIGxpbmUtaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xuICBib3gtc2hhZG93OiBub25lO1xuXG4gIGNvbG9yOiB2YXIoLS1qcC11aS1mb250LWNvbG9yMSk7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLWxheW91dC1jb2xvcjIpO1xuICBib3JkZXItY29sb3I6IHZhcigtLWpwLWJvcmRlci1jb2xvcjIpO1xuICBib3JkZXI6IG5vbmU7XG4gIHVzZXItc2VsZWN0OiBub25lO1xufVxuXG4uanVweXRlci1idXR0b24gaS5mYSB7XG4gIG1hcmdpbi1yaWdodDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtbWFyZ2luKTtcbiAgcG9pbnRlci1ldmVudHM6IG5vbmU7XG59XG5cbi5qdXB5dGVyLWJ1dHRvbjplbXB0eTpiZWZvcmUge1xuICBjb250ZW50OiAnXFwyMDBiJzsgLyogemVyby13aWR0aCBzcGFjZSAqL1xufVxuXG4uanVweXRlci13aWRnZXRzLmp1cHl0ZXItYnV0dG9uOmRpc2FibGVkIHtcbiAgb3BhY2l0eTogdmFyKC0tanAtd2lkZ2V0cy1kaXNhYmxlZC1vcGFjaXR5KTtcbn1cblxuLmp1cHl0ZXItYnV0dG9uIGkuZmEuY2VudGVyIHtcbiAgbWFyZ2luLXJpZ2h0OiAwO1xufVxuXG4uanVweXRlci1idXR0b246aG92ZXI6ZW5hYmxlZCxcbi5qdXB5dGVyLWJ1dHRvbjpmb2N1czplbmFibGVkIHtcbiAgLyogTUQgTGl0ZSAyZHAgc2hhZG93ICovXG4gIGJveC1zaGFkb3c6IDAgMnB4IDJweCAwIHJnYmEoMCwgMCwgMCwgdmFyKC0tbWQtc2hhZG93LWtleS1wZW51bWJyYS1vcGFjaXR5KSksXG4gICAgMCAzcHggMXB4IC0ycHggcmdiYSgwLCAwLCAwLCB2YXIoLS1tZC1zaGFkb3cta2V5LXVtYnJhLW9wYWNpdHkpKSxcbiAgICAwIDFweCA1cHggMCByZ2JhKDAsIDAsIDAsIHZhcigtLW1kLXNoYWRvdy1hbWJpZW50LXNoYWRvdy1vcGFjaXR5KSk7XG59XG5cbi5qdXB5dGVyLWJ1dHRvbjphY3RpdmUsXG4uanVweXRlci1idXR0b24ubW9kLWFjdGl2ZSB7XG4gIC8qIE1EIExpdGUgNGRwIHNoYWRvdyAqL1xuICBib3gtc2hhZG93OiAwIDRweCA1cHggMCByZ2JhKDAsIDAsIDAsIHZhcigtLW1kLXNoYWRvdy1rZXktcGVudW1icmEtb3BhY2l0eSkpLFxuICAgIDAgMXB4IDEwcHggMCByZ2JhKDAsIDAsIDAsIHZhcigtLW1kLXNoYWRvdy1hbWJpZW50LXNoYWRvdy1vcGFjaXR5KSksXG4gICAgMCAycHggNHB4IC0xcHggcmdiYSgwLCAwLCAwLCB2YXIoLS1tZC1zaGFkb3cta2V5LXVtYnJhLW9wYWNpdHkpKTtcbiAgY29sb3I6IHZhcigtLWpwLXVpLWZvbnQtY29sb3IxKTtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtbGF5b3V0LWNvbG9yMyk7XG59XG5cbi5qdXB5dGVyLWJ1dHRvbjpmb2N1czplbmFibGVkIHtcbiAgb3V0bGluZTogMXB4IHNvbGlkIHZhcigtLWpwLXdpZGdldHMtaW5wdXQtZm9jdXMtYm9yZGVyLWNvbG9yKTtcbn1cblxuLyogQnV0dG9uIFwiUHJpbWFyeVwiIFN0eWxpbmcgKi9cblxuLmp1cHl0ZXItYnV0dG9uLm1vZC1wcmltYXJ5IHtcbiAgY29sb3I6IHZhcigtLWpwLXVpLWludmVyc2UtZm9udC1jb2xvcjEpO1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC1icmFuZC1jb2xvcjEpO1xufVxuXG4uanVweXRlci1idXR0b24ubW9kLXByaW1hcnkubW9kLWFjdGl2ZSB7XG4gIGNvbG9yOiB2YXIoLS1qcC11aS1pbnZlcnNlLWZvbnQtY29sb3IwKTtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtYnJhbmQtY29sb3IwKTtcbn1cblxuLmp1cHl0ZXItYnV0dG9uLm1vZC1wcmltYXJ5OmFjdGl2ZSB7XG4gIGNvbG9yOiB2YXIoLS1qcC11aS1pbnZlcnNlLWZvbnQtY29sb3IwKTtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtYnJhbmQtY29sb3IwKTtcbn1cblxuLyogQnV0dG9uIFwiU3VjY2Vzc1wiIFN0eWxpbmcgKi9cblxuLmp1cHl0ZXItYnV0dG9uLm1vZC1zdWNjZXNzIHtcbiAgY29sb3I6IHZhcigtLWpwLXVpLWludmVyc2UtZm9udC1jb2xvcjEpO1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC1zdWNjZXNzLWNvbG9yMSk7XG59XG5cbi5qdXB5dGVyLWJ1dHRvbi5tb2Qtc3VjY2Vzcy5tb2QtYWN0aXZlIHtcbiAgY29sb3I6IHZhcigtLWpwLXVpLWludmVyc2UtZm9udC1jb2xvcjApO1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC1zdWNjZXNzLWNvbG9yMCk7XG59XG5cbi5qdXB5dGVyLWJ1dHRvbi5tb2Qtc3VjY2VzczphY3RpdmUge1xuICBjb2xvcjogdmFyKC0tanAtdWktaW52ZXJzZS1mb250LWNvbG9yMCk7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLXN1Y2Nlc3MtY29sb3IwKTtcbn1cblxuLyogQnV0dG9uIFwiSW5mb1wiIFN0eWxpbmcgKi9cblxuLmp1cHl0ZXItYnV0dG9uLm1vZC1pbmZvIHtcbiAgY29sb3I6IHZhcigtLWpwLXVpLWludmVyc2UtZm9udC1jb2xvcjEpO1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC1pbmZvLWNvbG9yMSk7XG59XG5cbi5qdXB5dGVyLWJ1dHRvbi5tb2QtaW5mby5tb2QtYWN0aXZlIHtcbiAgY29sb3I6IHZhcigtLWpwLXVpLWludmVyc2UtZm9udC1jb2xvcjApO1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC1pbmZvLWNvbG9yMCk7XG59XG5cbi5qdXB5dGVyLWJ1dHRvbi5tb2QtaW5mbzphY3RpdmUge1xuICBjb2xvcjogdmFyKC0tanAtdWktaW52ZXJzZS1mb250LWNvbG9yMCk7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLWluZm8tY29sb3IwKTtcbn1cblxuLyogQnV0dG9uIFwiV2FybmluZ1wiIFN0eWxpbmcgKi9cblxuLmp1cHl0ZXItYnV0dG9uLm1vZC13YXJuaW5nIHtcbiAgY29sb3I6IHZhcigtLWpwLXVpLWludmVyc2UtZm9udC1jb2xvcjEpO1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC13YXJuLWNvbG9yMSk7XG59XG5cbi5qdXB5dGVyLWJ1dHRvbi5tb2Qtd2FybmluZy5tb2QtYWN0aXZlIHtcbiAgY29sb3I6IHZhcigtLWpwLXVpLWludmVyc2UtZm9udC1jb2xvcjApO1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC13YXJuLWNvbG9yMCk7XG59XG5cbi5qdXB5dGVyLWJ1dHRvbi5tb2Qtd2FybmluZzphY3RpdmUge1xuICBjb2xvcjogdmFyKC0tanAtdWktaW52ZXJzZS1mb250LWNvbG9yMCk7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLXdhcm4tY29sb3IwKTtcbn1cblxuLyogQnV0dG9uIFwiRGFuZ2VyXCIgU3R5bGluZyAqL1xuXG4uanVweXRlci1idXR0b24ubW9kLWRhbmdlciB7XG4gIGNvbG9yOiB2YXIoLS1qcC11aS1pbnZlcnNlLWZvbnQtY29sb3IxKTtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtZXJyb3ItY29sb3IxKTtcbn1cblxuLmp1cHl0ZXItYnV0dG9uLm1vZC1kYW5nZXIubW9kLWFjdGl2ZSB7XG4gIGNvbG9yOiB2YXIoLS1qcC11aS1pbnZlcnNlLWZvbnQtY29sb3IwKTtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtZXJyb3ItY29sb3IwKTtcbn1cblxuLmp1cHl0ZXItYnV0dG9uLm1vZC1kYW5nZXI6YWN0aXZlIHtcbiAgY29sb3I6IHZhcigtLWpwLXVpLWludmVyc2UtZm9udC1jb2xvcjApO1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC1lcnJvci1jb2xvcjApO1xufVxuXG4vKiBXaWRnZXQgQnV0dG9uLCBXaWRnZXQgVG9nZ2xlIEJ1dHRvbiwgV2lkZ2V0IFVwbG9hZCAqL1xuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtYnV0dG9uLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi8gLndpZGdldC10b2dnbGUtYnV0dG9uLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi8gLndpZGdldC11cGxvYWQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1idXR0b24sXG4uanVweXRlci13aWRnZXQtdG9nZ2xlLWJ1dHRvbixcbi5qdXB5dGVyLXdpZGdldC11cGxvYWQge1xuICB3aWR0aDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtd2lkdGgtc2hvcnQpO1xufVxuXG4vKiBXaWRnZXQgTGFiZWwgU3R5bGluZyAqL1xuXG4vKiBPdmVycmlkZSBCb290c3RyYXAgbGFiZWwgY3NzICovXG4uanVweXRlci13aWRnZXRzIGxhYmVsIHtcbiAgbWFyZ2luLWJvdHRvbTogaW5pdGlhbDtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWxhYmVsLWJhc2ljLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtbGFiZWwtYmFzaWMge1xuICAvKiBCYXNpYyBMYWJlbCAqL1xuICBjb2xvcjogdmFyKC0tanAtd2lkZ2V0cy1sYWJlbC1jb2xvcik7XG4gIGZvbnQtc2l6ZTogdmFyKC0tanAtd2lkZ2V0cy1mb250LXNpemUpO1xuICBvdmVyZmxvdzogaGlkZGVuO1xuICB0ZXh0LW92ZXJmbG93OiBlbGxpcHNpcztcbiAgd2hpdGUtc3BhY2U6IG5vd3JhcDtcbiAgbGluZS1oZWlnaHQ6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLWhlaWdodCk7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1sYWJlbCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWxhYmVsIHtcbiAgLyogTGFiZWwgKi9cbiAgY29sb3I6IHZhcigtLWpwLXdpZGdldHMtbGFiZWwtY29sb3IpO1xuICBmb250LXNpemU6IHZhcigtLWpwLXdpZGdldHMtZm9udC1zaXplKTtcbiAgb3ZlcmZsb3c6IGhpZGRlbjtcbiAgdGV4dC1vdmVyZmxvdzogZWxsaXBzaXM7XG4gIHdoaXRlLXNwYWNlOiBub3dyYXA7XG4gIGxpbmUtaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtaW5saW5lLWhib3ggLndpZGdldC1sYWJlbCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWlubGluZS1oYm94IC5qdXB5dGVyLXdpZGdldC1sYWJlbCB7XG4gIC8qIEhvcml6b250YWwgV2lkZ2V0IExhYmVsICovXG4gIGNvbG9yOiB2YXIoLS1qcC13aWRnZXRzLWxhYmVsLWNvbG9yKTtcbiAgdGV4dC1hbGlnbjogcmlnaHQ7XG4gIG1hcmdpbi1yaWdodDogY2FsYyh2YXIoLS1qcC13aWRnZXRzLWlubGluZS1tYXJnaW4pICogMik7XG4gIHdpZHRoOiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1sYWJlbC13aWR0aCk7XG4gIGZsZXgtc2hyaW5rOiAwO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtaW5saW5lLXZib3ggLndpZGdldC1sYWJlbCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWlubGluZS12Ym94IC5qdXB5dGVyLXdpZGdldC1sYWJlbCB7XG4gIC8qIFZlcnRpY2FsIFdpZGdldCBMYWJlbCAqL1xuICBjb2xvcjogdmFyKC0tanAtd2lkZ2V0cy1sYWJlbC1jb2xvcik7XG4gIHRleHQtYWxpZ246IGNlbnRlcjtcbiAgbGluZS1oZWlnaHQ6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLWhlaWdodCk7XG59XG5cbi8qIFdpZGdldCBSZWFkb3V0IFN0eWxpbmcgKi9cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LXJlYWRvdXQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1yZWFkb3V0IHtcbiAgY29sb3I6IHZhcigtLWpwLXdpZGdldHMtcmVhZG91dC1jb2xvcik7XG4gIGZvbnQtc2l6ZTogdmFyKC0tanAtd2lkZ2V0cy1mb250LXNpemUpO1xuICBoZWlnaHQ6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLWhlaWdodCk7XG4gIGxpbmUtaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xuICBvdmVyZmxvdzogaGlkZGVuO1xuICB3aGl0ZS1zcGFjZTogbm93cmFwO1xuICB0ZXh0LWFsaWduOiBjZW50ZXI7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1yZWFkb3V0Lm92ZXJmbG93LCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtcmVhZG91dC5vdmVyZmxvdyB7XG4gIC8qIE92ZXJmbG93aW5nIFJlYWRvdXQgKi9cblxuICAvKiBGcm9tIE1hdGVyaWFsIERlc2lnbiBMaXRlXG4gICAgICAgIHNoYWRvdy1rZXktdW1icmEtb3BhY2l0eTogMC4yO1xuICAgICAgICBzaGFkb3cta2V5LXBlbnVtYnJhLW9wYWNpdHk6IDAuMTQ7XG4gICAgICAgIHNoYWRvdy1hbWJpZW50LXNoYWRvdy1vcGFjaXR5OiAwLjEyO1xuICAgICAqL1xuICAtd2Via2l0LWJveC1zaGFkb3c6IDAgMnB4IDJweCAwIHJnYmEoMCwgMCwgMCwgMC4yKSxcbiAgICAwIDNweCAxcHggLTJweCByZ2JhKDAsIDAsIDAsIDAuMTQpLCAwIDFweCA1cHggMCByZ2JhKDAsIDAsIDAsIDAuMTIpO1xuXG4gIC1tb3otYm94LXNoYWRvdzogMCAycHggMnB4IDAgcmdiYSgwLCAwLCAwLCAwLjIpLFxuICAgIDAgM3B4IDFweCAtMnB4IHJnYmEoMCwgMCwgMCwgMC4xNCksIDAgMXB4IDVweCAwIHJnYmEoMCwgMCwgMCwgMC4xMik7XG5cbiAgYm94LXNoYWRvdzogMCAycHggMnB4IDAgcmdiYSgwLCAwLCAwLCAwLjIpLCAwIDNweCAxcHggLTJweCByZ2JhKDAsIDAsIDAsIDAuMTQpLFxuICAgIDAgMXB4IDVweCAwIHJnYmEoMCwgMCwgMCwgMC4xMik7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1pbmxpbmUtaGJveCAud2lkZ2V0LXJlYWRvdXQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1pbmxpbmUtaGJveCAuanVweXRlci13aWRnZXQtcmVhZG91dCB7XG4gIC8qIEhvcml6b250YWwgUmVhZG91dCAqL1xuICB0ZXh0LWFsaWduOiBjZW50ZXI7XG4gIG1heC13aWR0aDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtd2lkdGgtc2hvcnQpO1xuICBtaW4td2lkdGg6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLXdpZHRoLXRpbnkpO1xuICBtYXJnaW4tbGVmdDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtbWFyZ2luKTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWlubGluZS12Ym94IC53aWRnZXQtcmVhZG91dCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWlubGluZS12Ym94IC5qdXB5dGVyLXdpZGdldC1yZWFkb3V0IHtcbiAgLyogVmVydGljYWwgUmVhZG91dCAqL1xuICBtYXJnaW4tdG9wOiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1tYXJnaW4pO1xuICAvKiBhcyB3aWRlIGFzIHRoZSB3aWRnZXQgKi9cbiAgd2lkdGg6IGluaGVyaXQ7XG59XG5cbi8qIFdpZGdldCBDaGVja2JveCBTdHlsaW5nICovXG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1jaGVja2JveCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWNoZWNrYm94IHtcbiAgd2lkdGg6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLXdpZHRoKTtcbiAgaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xuICBsaW5lLWhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWNoZWNrYm94IGlucHV0W3R5cGU9J2NoZWNrYm94J10sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1jaGVja2JveCBpbnB1dFt0eXBlPSdjaGVja2JveCddIHtcbiAgbWFyZ2luOiAwcHggY2FsYyh2YXIoLS1qcC13aWRnZXRzLWlubGluZS1tYXJnaW4pICogMikgMHB4IDBweDtcbiAgbGluZS1oZWlnaHQ6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLWhlaWdodCk7XG4gIGZvbnQtc2l6ZTogbGFyZ2U7XG4gIGZsZXgtZ3JvdzogMTtcbiAgZmxleC1zaHJpbms6IDA7XG4gIGFsaWduLXNlbGY6IGNlbnRlcjtcbn1cblxuLyogV2lkZ2V0IFZhbGlkIFN0eWxpbmcgKi9cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LXZhbGlkLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtdmFsaWQge1xuICBoZWlnaHQ6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLWhlaWdodCk7XG4gIGxpbmUtaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xuICB3aWR0aDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtd2lkdGgtc2hvcnQpO1xuICBmb250LXNpemU6IHZhcigtLWpwLXdpZGdldHMtZm9udC1zaXplKTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LXZhbGlkIGksIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC12YWxpZCBpIHtcbiAgbGluZS1oZWlnaHQ6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLWhlaWdodCk7XG4gIG1hcmdpbi1yaWdodDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtbWFyZ2luKTtcbiAgbWFyZ2luLWxlZnQ6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLW1hcmdpbik7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC12YWxpZC5tb2QtdmFsaWQgaSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LXZhbGlkLm1vZC12YWxpZCBpIHtcbiAgY29sb3I6IGdyZWVuO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtdmFsaWQubW9kLWludmFsaWQgaSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LXZhbGlkLm1vZC1pbnZhbGlkIGkge1xuICBjb2xvcjogcmVkO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtdmFsaWQubW9kLXZhbGlkIC53aWRnZXQtdmFsaWQtcmVhZG91dCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LXZhbGlkLm1vZC12YWxpZCAuanVweXRlci13aWRnZXQtdmFsaWQtcmVhZG91dCB7XG4gIGRpc3BsYXk6IG5vbmU7XG59XG5cbi8qIFdpZGdldCBUZXh0IGFuZCBUZXh0QXJlYSBTdHlsaW5nICovXG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC10ZXh0YXJlYSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovIC53aWRnZXQtdGV4dCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LXRleHRhcmVhLFxuLmp1cHl0ZXItd2lkZ2V0LXRleHQge1xuICB3aWR0aDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtd2lkdGgpO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtdGV4dCBpbnB1dFt0eXBlPSd0ZXh0J10sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLyAud2lkZ2V0LXRleHQgaW5wdXRbdHlwZT0nbnVtYmVyJ10sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLyAud2lkZ2V0LXRleHQgaW5wdXRbdHlwZT0ncGFzc3dvcmQnXSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LXRleHQgaW5wdXRbdHlwZT0ndGV4dCddLFxuLmp1cHl0ZXItd2lkZ2V0LXRleHQgaW5wdXRbdHlwZT0nbnVtYmVyJ10sXG4uanVweXRlci13aWRnZXQtdGV4dCBpbnB1dFt0eXBlPSdwYXNzd29yZCddIHtcbiAgaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtdGV4dCBpbnB1dFt0eXBlPSd0ZXh0J106ZGlzYWJsZWQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLyAud2lkZ2V0LXRleHQgaW5wdXRbdHlwZT0nbnVtYmVyJ106ZGlzYWJsZWQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLyAud2lkZ2V0LXRleHQgaW5wdXRbdHlwZT0ncGFzc3dvcmQnXTpkaXNhYmxlZCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovIC53aWRnZXQtdGV4dGFyZWEgdGV4dGFyZWE6ZGlzYWJsZWQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC10ZXh0IGlucHV0W3R5cGU9J3RleHQnXTpkaXNhYmxlZCxcbi5qdXB5dGVyLXdpZGdldC10ZXh0IGlucHV0W3R5cGU9J251bWJlciddOmRpc2FibGVkLFxuLmp1cHl0ZXItd2lkZ2V0LXRleHQgaW5wdXRbdHlwZT0ncGFzc3dvcmQnXTpkaXNhYmxlZCxcbi5qdXB5dGVyLXdpZGdldC10ZXh0YXJlYSB0ZXh0YXJlYTpkaXNhYmxlZCB7XG4gIG9wYWNpdHk6IHZhcigtLWpwLXdpZGdldHMtZGlzYWJsZWQtb3BhY2l0eSk7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC10ZXh0IGlucHV0W3R5cGU9J3RleHQnXSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovIC53aWRnZXQtdGV4dCBpbnB1dFt0eXBlPSdudW1iZXInXSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovIC53aWRnZXQtdGV4dCBpbnB1dFt0eXBlPSdwYXNzd29yZCddLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi8gLndpZGdldC10ZXh0YXJlYSB0ZXh0YXJlYSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LXRleHQgaW5wdXRbdHlwZT0ndGV4dCddLFxuLmp1cHl0ZXItd2lkZ2V0LXRleHQgaW5wdXRbdHlwZT0nbnVtYmVyJ10sXG4uanVweXRlci13aWRnZXQtdGV4dCBpbnB1dFt0eXBlPSdwYXNzd29yZCddLFxuLmp1cHl0ZXItd2lkZ2V0LXRleHRhcmVhIHRleHRhcmVhIHtcbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcbiAgYm9yZGVyOiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LWJvcmRlci13aWR0aCkgc29saWRcbiAgICB2YXIoLS1qcC13aWRnZXRzLWlucHV0LWJvcmRlci1jb2xvcik7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLXdpZGdldHMtaW5wdXQtYmFja2dyb3VuZC1jb2xvcik7XG4gIGNvbG9yOiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LWNvbG9yKTtcbiAgZm9udC1zaXplOiB2YXIoLS1qcC13aWRnZXRzLWZvbnQtc2l6ZSk7XG4gIGZsZXgtZ3JvdzogMTtcbiAgbWluLXdpZHRoOiAwOyAvKiBUaGlzIG1ha2VzIGl0IHBvc3NpYmxlIGZvciB0aGUgZmxleGJveCB0byBzaHJpbmsgdGhpcyBpbnB1dCAqL1xuICBmbGV4LXNocmluazogMTtcbiAgb3V0bGluZTogbm9uZSAhaW1wb3J0YW50O1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtdGV4dCBpbnB1dFt0eXBlPSd0ZXh0J10sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLyAud2lkZ2V0LXRleHQgaW5wdXRbdHlwZT0ncGFzc3dvcmQnXSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovIC53aWRnZXQtdGV4dGFyZWEgdGV4dGFyZWEsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC10ZXh0IGlucHV0W3R5cGU9J3RleHQnXSxcbi5qdXB5dGVyLXdpZGdldC10ZXh0IGlucHV0W3R5cGU9J3Bhc3N3b3JkJ10sXG4uanVweXRlci13aWRnZXQtdGV4dGFyZWEgdGV4dGFyZWEge1xuICBwYWRkaW5nOiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LXBhZGRpbmcpXG4gICAgY2FsYyh2YXIoLS1qcC13aWRnZXRzLWlucHV0LXBhZGRpbmcpICogMik7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC10ZXh0IGlucHV0W3R5cGU9J251bWJlciddLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtdGV4dCBpbnB1dFt0eXBlPSdudW1iZXInXSB7XG4gIHBhZGRpbmc6IHZhcigtLWpwLXdpZGdldHMtaW5wdXQtcGFkZGluZykgMCB2YXIoLS1qcC13aWRnZXRzLWlucHV0LXBhZGRpbmcpXG4gICAgY2FsYyh2YXIoLS1qcC13aWRnZXRzLWlucHV0LXBhZGRpbmcpICogMik7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC10ZXh0YXJlYSB0ZXh0YXJlYSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LXRleHRhcmVhIHRleHRhcmVhIHtcbiAgaGVpZ2h0OiBpbmhlcml0O1xuICB3aWR0aDogaW5oZXJpdDtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LXRleHQgaW5wdXQ6Zm9jdXMsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLyAud2lkZ2V0LXRleHRhcmVhIHRleHRhcmVhOmZvY3VzLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtdGV4dCBpbnB1dDpmb2N1cyxcbi5qdXB5dGVyLXdpZGdldC10ZXh0YXJlYSB0ZXh0YXJlYTpmb2N1cyB7XG4gIGJvcmRlci1jb2xvcjogdmFyKC0tanAtd2lkZ2V0cy1pbnB1dC1mb2N1cy1ib3JkZXItY29sb3IpO1xufVxuXG4vKiBIb3Jpem9udGFsIFNsaWRlciAqL1xuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWhzbGlkZXIsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1oc2xpZGVyIHtcbiAgd2lkdGg6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLXdpZHRoKTtcbiAgaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xuICBsaW5lLWhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KTtcblxuICAvKiBPdmVycmlkZSB0aGUgYWxpZ24taXRlbXMgYmFzZWxpbmUuIFRoaXMgd2F5LCB0aGUgZGVzY3JpcHRpb24gYW5kIHJlYWRvdXRcbiAgICBzdGlsbCBzZWVtIHRvIGFsaWduIHRoZWlyIGJhc2VsaW5lIHByb3Blcmx5LCBhbmQgd2UgZG9uJ3QgaGF2ZSB0byBoYXZlXG4gICAgYWxpZ24tc2VsZjogc3RyZXRjaCBpbiB0aGUgLnNsaWRlci1jb250YWluZXIuICovXG4gIGFsaWduLWl0ZW1zOiBjZW50ZXI7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldHMtc2xpZGVyIC5zbGlkZXItY29udGFpbmVyLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLXNsaWRlciAuc2xpZGVyLWNvbnRhaW5lciB7XG4gIG92ZXJmbG93OiB2aXNpYmxlO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtaHNsaWRlciAuc2xpZGVyLWNvbnRhaW5lciwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWhzbGlkZXIgLnNsaWRlci1jb250YWluZXIge1xuICBtYXJnaW4tbGVmdDogY2FsYyhcbiAgICB2YXIoLS1qcC13aWRnZXRzLXNsaWRlci1oYW5kbGUtc2l6ZSkgLyAyIC0gMiAqXG4gICAgICB2YXIoLS1qcC13aWRnZXRzLXNsaWRlci1ib3JkZXItd2lkdGgpXG4gICk7XG4gIG1hcmdpbi1yaWdodDogY2FsYyhcbiAgICB2YXIoLS1qcC13aWRnZXRzLXNsaWRlci1oYW5kbGUtc2l6ZSkgLyAyIC0gMiAqXG4gICAgICB2YXIoLS1qcC13aWRnZXRzLXNsaWRlci1ib3JkZXItd2lkdGgpXG4gICk7XG4gIGZsZXg6IDEgMSB2YXIoLS1qcC13aWRnZXRzLWlubGluZS13aWR0aC1zaG9ydCk7XG59XG5cbi8qIFZlcnRpY2FsIFNsaWRlciAqL1xuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtdmJveCAud2lkZ2V0LWxhYmVsLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtdmJveCAuanVweXRlci13aWRnZXQtbGFiZWwge1xuICBoZWlnaHQ6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLWhlaWdodCk7XG4gIGxpbmUtaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtdnNsaWRlciwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LXZzbGlkZXIge1xuICAvKiBWZXJ0aWNhbCBTbGlkZXIgKi9cbiAgaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLXZlcnRpY2FsLWhlaWdodCk7XG4gIHdpZHRoOiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS13aWR0aC10aW55KTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LXZzbGlkZXIgLnNsaWRlci1jb250YWluZXIsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC12c2xpZGVyIC5zbGlkZXItY29udGFpbmVyIHtcbiAgZmxleDogMSAxIHZhcigtLWpwLXdpZGdldHMtaW5saW5lLXdpZHRoLXNob3J0KTtcbiAgbWFyZ2luLWxlZnQ6IGF1dG87XG4gIG1hcmdpbi1yaWdodDogYXV0bztcbiAgbWFyZ2luLWJvdHRvbTogY2FsYyhcbiAgICB2YXIoLS1qcC13aWRnZXRzLXNsaWRlci1oYW5kbGUtc2l6ZSkgLyAyIC0gMiAqXG4gICAgICB2YXIoLS1qcC13aWRnZXRzLXNsaWRlci1ib3JkZXItd2lkdGgpXG4gICk7XG4gIG1hcmdpbi10b3A6IGNhbGMoXG4gICAgdmFyKC0tanAtd2lkZ2V0cy1zbGlkZXItaGFuZGxlLXNpemUpIC8gMiAtIDIgKlxuICAgICAgdmFyKC0tanAtd2lkZ2V0cy1zbGlkZXItYm9yZGVyLXdpZHRoKVxuICApO1xuICBkaXNwbGF5OiBmbGV4O1xuICBmbGV4LWRpcmVjdGlvbjogY29sdW1uO1xufVxuXG4vKiBXaWRnZXQgUHJvZ3Jlc3MgU3R5bGluZyAqL1xuXG4ucHJvZ3Jlc3MtYmFyIHtcbiAgLXdlYmtpdC10cmFuc2l0aW9uOiBub25lO1xuICAtbW96LXRyYW5zaXRpb246IG5vbmU7XG4gIC1tcy10cmFuc2l0aW9uOiBub25lO1xuICAtby10cmFuc2l0aW9uOiBub25lO1xuICB0cmFuc2l0aW9uOiBub25lO1xufVxuXG4ucHJvZ3Jlc3MtYmFyIHtcbiAgaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xufVxuXG4ucHJvZ3Jlc3MtYmFyIHtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtYnJhbmQtY29sb3IxKTtcbn1cblxuLnByb2dyZXNzLWJhci1zdWNjZXNzIHtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtc3VjY2Vzcy1jb2xvcjEpO1xufVxuXG4ucHJvZ3Jlc3MtYmFyLWluZm8ge1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC1pbmZvLWNvbG9yMSk7XG59XG5cbi5wcm9ncmVzcy1iYXItd2FybmluZyB7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLXdhcm4tY29sb3IxKTtcbn1cblxuLnByb2dyZXNzLWJhci1kYW5nZXIge1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC1lcnJvci1jb2xvcjEpO1xufVxuXG4ucHJvZ3Jlc3Mge1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC1sYXlvdXQtY29sb3IyKTtcbiAgYm9yZGVyOiBub25lO1xuICBib3gtc2hhZG93OiBub25lO1xufVxuXG4vKiBIb3Jpc29udGFsIFByb2dyZXNzICovXG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1ocHJvZ3Jlc3MsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1ocHJvZ3Jlc3Mge1xuICAvKiBQcm9ncmVzcyBCYXIgKi9cbiAgaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xuICBsaW5lLWhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KTtcbiAgd2lkdGg6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLXdpZHRoKTtcbiAgYWxpZ24taXRlbXM6IGNlbnRlcjtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWhwcm9ncmVzcyAucHJvZ3Jlc3MsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1ocHJvZ3Jlc3MgLnByb2dyZXNzIHtcbiAgZmxleC1ncm93OiAxO1xuICBtYXJnaW4tdG9wOiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LXBhZGRpbmcpO1xuICBtYXJnaW4tYm90dG9tOiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LXBhZGRpbmcpO1xuICBhbGlnbi1zZWxmOiBzdHJldGNoO1xuICAvKiBPdmVycmlkZSBib290c3RyYXAgc3R5bGUgKi9cbiAgaGVpZ2h0OiBpbml0aWFsO1xufVxuXG4vKiBWZXJ0aWNhbCBQcm9ncmVzcyAqL1xuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtdnByb2dyZXNzLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtdnByb2dyZXNzIHtcbiAgaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLXZlcnRpY2FsLWhlaWdodCk7XG4gIHdpZHRoOiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS13aWR0aC10aW55KTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LXZwcm9ncmVzcyAucHJvZ3Jlc3MsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC12cHJvZ3Jlc3MgLnByb2dyZXNzIHtcbiAgZmxleC1ncm93OiAxO1xuICB3aWR0aDogdmFyKC0tanAtd2lkZ2V0cy1wcm9ncmVzcy10aGlja25lc3MpO1xuICBtYXJnaW4tbGVmdDogYXV0bztcbiAgbWFyZ2luLXJpZ2h0OiBhdXRvO1xuICBtYXJnaW4tYm90dG9tOiAwO1xufVxuXG4vKiBTZWxlY3QgV2lkZ2V0IFN0eWxpbmcgKi9cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWRyb3Bkb3duLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtZHJvcGRvd24ge1xuICBoZWlnaHQ6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLWhlaWdodCk7XG4gIHdpZHRoOiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS13aWR0aCk7XG4gIGxpbmUtaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtZHJvcGRvd24gPiBzZWxlY3QsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1kcm9wZG93biA+IHNlbGVjdCB7XG4gIHBhZGRpbmctcmlnaHQ6IDIwcHg7XG4gIGJvcmRlcjogdmFyKC0tanAtd2lkZ2V0cy1pbnB1dC1ib3JkZXItd2lkdGgpIHNvbGlkXG4gICAgdmFyKC0tanAtd2lkZ2V0cy1pbnB1dC1ib3JkZXItY29sb3IpO1xuICBib3JkZXItcmFkaXVzOiAwO1xuICBoZWlnaHQ6IGluaGVyaXQ7XG4gIGZsZXg6IDEgMSB2YXIoLS1qcC13aWRnZXRzLWlubGluZS13aWR0aC1zaG9ydCk7XG4gIG1pbi13aWR0aDogMDsgLyogVGhpcyBtYWtlcyBpdCBwb3NzaWJsZSBmb3IgdGhlIGZsZXhib3ggdG8gc2hyaW5rIHRoaXMgaW5wdXQgKi9cbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcbiAgb3V0bGluZTogbm9uZSAhaW1wb3J0YW50O1xuICBib3gtc2hhZG93OiBub25lO1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LWJhY2tncm91bmQtY29sb3IpO1xuICBjb2xvcjogdmFyKC0tanAtd2lkZ2V0cy1pbnB1dC1jb2xvcik7XG4gIGZvbnQtc2l6ZTogdmFyKC0tanAtd2lkZ2V0cy1mb250LXNpemUpO1xuICB2ZXJ0aWNhbC1hbGlnbjogdG9wO1xuICBwYWRkaW5nLWxlZnQ6IGNhbGModmFyKC0tanAtd2lkZ2V0cy1pbnB1dC1wYWRkaW5nKSAqIDIpO1xuICBhcHBlYXJhbmNlOiBub25lO1xuICAtd2Via2l0LWFwcGVhcmFuY2U6IG5vbmU7XG4gIC1tb3otYXBwZWFyYW5jZTogbm9uZTtcbiAgYmFja2dyb3VuZC1yZXBlYXQ6IG5vLXJlcGVhdDtcbiAgYmFja2dyb3VuZC1zaXplOiAyMHB4O1xuICBiYWNrZ3JvdW5kLXBvc2l0aW9uOiByaWdodCBjZW50ZXI7XG4gIGJhY2tncm91bmQtaW1hZ2U6IHZhcigtLWpwLXdpZGdldHMtZHJvcGRvd24tYXJyb3cpO1xufVxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWRyb3Bkb3duID4gc2VsZWN0OmZvY3VzLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtZHJvcGRvd24gPiBzZWxlY3Q6Zm9jdXMge1xuICBib3JkZXItY29sb3I6IHZhcigtLWpwLXdpZGdldHMtaW5wdXQtZm9jdXMtYm9yZGVyLWNvbG9yKTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWRyb3Bkb3duID4gc2VsZWN0OmRpc2FibGVkLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtZHJvcGRvd24gPiBzZWxlY3Q6ZGlzYWJsZWQge1xuICBvcGFjaXR5OiB2YXIoLS1qcC13aWRnZXRzLWRpc2FibGVkLW9wYWNpdHkpO1xufVxuXG4vKiBUbyBkaXNhYmxlIHRoZSBkb3R0ZWQgYm9yZGVyIGluIEZpcmVmb3ggYXJvdW5kIHNlbGVjdCBjb250cm9scy5cbiAgIFNlZSBodHRwOi8vc3RhY2tvdmVyZmxvdy5jb20vYS8xODg1MzAwMiAqL1xuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWRyb3Bkb3duID4gc2VsZWN0Oi1tb3otZm9jdXNyaW5nLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtZHJvcGRvd24gPiBzZWxlY3Q6LW1vei1mb2N1c3Jpbmcge1xuICBjb2xvcjogdHJhbnNwYXJlbnQ7XG4gIHRleHQtc2hhZG93OiAwIDAgMCAjMDAwO1xufVxuXG4vKiBTZWxlY3QgYW5kIFNlbGVjdE11bHRpcGxlICovXG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1zZWxlY3QsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1zZWxlY3Qge1xuICB3aWR0aDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtd2lkdGgpO1xuICBsaW5lLWhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KTtcblxuICAvKiBCZWNhdXNlIEZpcmVmb3ggZGVmaW5lcyB0aGUgYmFzZWxpbmUgb2YgYSBzZWxlY3QgYXMgdGhlIGJvdHRvbSBvZiB0aGVcbiAgICBjb250cm9sLCB3ZSBhbGlnbiB0aGUgZW50aXJlIGNvbnRyb2wgdG8gdGhlIHRvcCBhbmQgYWRkIHBhZGRpbmcgdG8gdGhlXG4gICAgc2VsZWN0IHRvIGdldCBhbiBhcHByb3hpbWF0ZSBmaXJzdCBsaW5lIGJhc2VsaW5lIGFsaWdubWVudC4gKi9cbiAgYWxpZ24taXRlbXM6IGZsZXgtc3RhcnQ7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1zZWxlY3QgPiBzZWxlY3QsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1zZWxlY3QgPiBzZWxlY3Qge1xuICBib3JkZXI6IHZhcigtLWpwLXdpZGdldHMtaW5wdXQtYm9yZGVyLXdpZHRoKSBzb2xpZFxuICAgIHZhcigtLWpwLXdpZGdldHMtaW5wdXQtYm9yZGVyLWNvbG9yKTtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtd2lkZ2V0cy1pbnB1dC1iYWNrZ3JvdW5kLWNvbG9yKTtcbiAgY29sb3I6IHZhcigtLWpwLXdpZGdldHMtaW5wdXQtY29sb3IpO1xuICBmb250LXNpemU6IHZhcigtLWpwLXdpZGdldHMtZm9udC1zaXplKTtcbiAgZmxleDogMSAxIHZhcigtLWpwLXdpZGdldHMtaW5saW5lLXdpZHRoLXNob3J0KTtcbiAgb3V0bGluZTogbm9uZSAhaW1wb3J0YW50O1xuICBvdmVyZmxvdzogYXV0bztcbiAgaGVpZ2h0OiBpbmhlcml0O1xuXG4gIC8qIEJlY2F1c2UgRmlyZWZveCBkZWZpbmVzIHRoZSBiYXNlbGluZSBvZiBhIHNlbGVjdCBhcyB0aGUgYm90dG9tIG9mIHRoZVxuICAgIGNvbnRyb2wsIHdlIGFsaWduIHRoZSBlbnRpcmUgY29udHJvbCB0byB0aGUgdG9wIGFuZCBhZGQgcGFkZGluZyB0byB0aGVcbiAgICBzZWxlY3QgdG8gZ2V0IGFuIGFwcHJveGltYXRlIGZpcnN0IGxpbmUgYmFzZWxpbmUgYWxpZ25tZW50LiAqL1xuICBwYWRkaW5nLXRvcDogNXB4O1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtc2VsZWN0ID4gc2VsZWN0OmZvY3VzLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtc2VsZWN0ID4gc2VsZWN0OmZvY3VzIHtcbiAgYm9yZGVyLWNvbG9yOiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LWZvY3VzLWJvcmRlci1jb2xvcik7XG59XG5cbi53aWdldC1zZWxlY3QgPiBzZWxlY3QgPiBvcHRpb24sXG4uanVweXRlci13aWdldC1zZWxlY3QgPiBzZWxlY3QgPiBvcHRpb24ge1xuICBwYWRkaW5nLWxlZnQ6IHZhcigtLWpwLXdpZGdldHMtaW5wdXQtcGFkZGluZyk7XG4gIGxpbmUtaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xuICAvKiBsaW5lLWhlaWdodCBkb2Vzbid0IHdvcmsgb24gc29tZSBicm93c2VycyBmb3Igc2VsZWN0IG9wdGlvbnMgKi9cbiAgcGFkZGluZy10b3A6IGNhbGMoXG4gICAgdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KSAtIHZhcigtLWpwLXdpZGdldHMtZm9udC1zaXplKSAvIDJcbiAgKTtcbiAgcGFkZGluZy1ib3R0b206IGNhbGMoXG4gICAgdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KSAtIHZhcigtLWpwLXdpZGdldHMtZm9udC1zaXplKSAvIDJcbiAgKTtcbn1cblxuLyogVG9nZ2xlIEJ1dHRvbnMgU3R5bGluZyAqL1xuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtdG9nZ2xlLWJ1dHRvbnMsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC10b2dnbGUtYnV0dG9ucyB7XG4gIGxpbmUtaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtdG9nZ2xlLWJ1dHRvbnMgLndpZGdldC10b2dnbGUtYnV0dG9uLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtdG9nZ2xlLWJ1dHRvbnMgLmp1cHl0ZXItd2lkZ2V0LXRvZ2dsZS1idXR0b24ge1xuICBtYXJnaW4tbGVmdDogdmFyKC0tanAtd2lkZ2V0cy1tYXJnaW4pO1xuICBtYXJnaW4tcmlnaHQ6IHZhcigtLWpwLXdpZGdldHMtbWFyZ2luKTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LXRvZ2dsZS1idXR0b25zIC5qdXB5dGVyLWJ1dHRvbjpkaXNhYmxlZCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LXRvZ2dsZS1idXR0b25zIC5qdXB5dGVyLWJ1dHRvbjpkaXNhYmxlZCB7XG4gIG9wYWNpdHk6IHZhcigtLWpwLXdpZGdldHMtZGlzYWJsZWQtb3BhY2l0eSk7XG59XG5cbi8qIFJhZGlvIEJ1dHRvbnMgU3R5bGluZyAqL1xuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtcmFkaW8sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1yYWRpbyB7XG4gIHdpZHRoOiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS13aWR0aCk7XG4gIGxpbmUtaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtcmFkaW8tYm94LCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtcmFkaW8tYm94IHtcbiAgZGlzcGxheTogZmxleDtcbiAgZmxleC1kaXJlY3Rpb246IGNvbHVtbjtcbiAgYWxpZ24taXRlbXM6IHN0cmV0Y2g7XG4gIGJveC1zaXppbmc6IGJvcmRlci1ib3g7XG4gIGZsZXgtZ3JvdzogMTtcbiAgbWFyZ2luLWJvdHRvbTogdmFyKC0tanAtd2lkZ2V0cy1yYWRpby1pdGVtLWhlaWdodC1hZGp1c3RtZW50KTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LXJhZGlvLWJveCBsYWJlbCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LXJhZGlvLWJveCBsYWJlbCB7XG4gIGhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1yYWRpby1pdGVtLWhlaWdodCk7XG4gIGxpbmUtaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLXJhZGlvLWl0ZW0taGVpZ2h0KTtcbiAgZm9udC1zaXplOiB2YXIoLS1qcC13aWRnZXRzLWZvbnQtc2l6ZSk7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1yYWRpby1ib3ggaW5wdXQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1yYWRpby1ib3ggaW5wdXQge1xuICBoZWlnaHQ6IHZhcigtLWpwLXdpZGdldHMtcmFkaW8taXRlbS1oZWlnaHQpO1xuICBsaW5lLWhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1yYWRpby1pdGVtLWhlaWdodCk7XG4gIG1hcmdpbjogMCBjYWxjKHZhcigtLWpwLXdpZGdldHMtaW5wdXQtcGFkZGluZykgKiAyKSAwIDFweDtcbiAgZmxvYXQ6IGxlZnQ7XG59XG5cbi8qIENvbG9yIFBpY2tlciBTdHlsaW5nICovXG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1jb2xvcnBpY2tlciwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWNvbG9ycGlja2VyIHtcbiAgd2lkdGg6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLXdpZHRoKTtcbiAgaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS1oZWlnaHQpO1xuICBsaW5lLWhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWNvbG9ycGlja2VyID4gLndpZGdldC1jb2xvcnBpY2tlci1pbnB1dCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWNvbG9ycGlja2VyID4gLmp1cHl0ZXItd2lkZ2V0LWNvbG9ycGlja2VyLWlucHV0IHtcbiAgZmxleC1ncm93OiAxO1xuICBmbGV4LXNocmluazogMTtcbiAgbWluLXdpZHRoOiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS13aWR0aC10aW55KTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWNvbG9ycGlja2VyIGlucHV0W3R5cGU9J2NvbG9yJ10sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1jb2xvcnBpY2tlciBpbnB1dFt0eXBlPSdjb2xvciddIHtcbiAgd2lkdGg6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLWhlaWdodCk7XG4gIGhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KTtcbiAgcGFkZGluZzogMCAycHg7IC8qIG1ha2UgdGhlIGNvbG9yIHNxdWFyZSBhY3R1YWxseSBzcXVhcmUgb24gQ2hyb21lIG9uIE9TIFggKi9cbiAgYmFja2dyb3VuZDogdmFyKC0tanAtd2lkZ2V0cy1pbnB1dC1iYWNrZ3JvdW5kLWNvbG9yKTtcbiAgY29sb3I6IHZhcigtLWpwLXdpZGdldHMtaW5wdXQtY29sb3IpO1xuICBib3JkZXI6IHZhcigtLWpwLXdpZGdldHMtaW5wdXQtYm9yZGVyLXdpZHRoKSBzb2xpZFxuICAgIHZhcigtLWpwLXdpZGdldHMtaW5wdXQtYm9yZGVyLWNvbG9yKTtcbiAgYm9yZGVyLWxlZnQ6IG5vbmU7XG4gIGZsZXgtZ3JvdzogMDtcbiAgZmxleC1zaHJpbms6IDA7XG4gIGJveC1zaXppbmc6IGJvcmRlci1ib3g7XG4gIGFsaWduLXNlbGY6IHN0cmV0Y2g7XG4gIG91dGxpbmU6IG5vbmUgIWltcG9ydGFudDtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWNvbG9ycGlja2VyLmNvbmNpc2UgaW5wdXRbdHlwZT0nY29sb3InXSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWNvbG9ycGlja2VyLmNvbmNpc2UgaW5wdXRbdHlwZT0nY29sb3InXSB7XG4gIGJvcmRlci1sZWZ0OiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LWJvcmRlci13aWR0aCkgc29saWRcbiAgICB2YXIoLS1qcC13aWRnZXRzLWlucHV0LWJvcmRlci1jb2xvcik7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1jb2xvcnBpY2tlciBpbnB1dFt0eXBlPSdjb2xvciddOmZvY3VzLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi8gLndpZGdldC1jb2xvcnBpY2tlciBpbnB1dFt0eXBlPSd0ZXh0J106Zm9jdXMsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1jb2xvcnBpY2tlciBpbnB1dFt0eXBlPSdjb2xvciddOmZvY3VzLFxuLmp1cHl0ZXItd2lkZ2V0LWNvbG9ycGlja2VyIGlucHV0W3R5cGU9J3RleHQnXTpmb2N1cyB7XG4gIGJvcmRlci1jb2xvcjogdmFyKC0tanAtd2lkZ2V0cy1pbnB1dC1mb2N1cy1ib3JkZXItY29sb3IpO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi53aWRnZXQtY29sb3JwaWNrZXIgaW5wdXRbdHlwZT0ndGV4dCddLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtY29sb3JwaWNrZXIgaW5wdXRbdHlwZT0ndGV4dCddIHtcbiAgZmxleC1ncm93OiAxO1xuICBvdXRsaW5lOiBub25lICFpbXBvcnRhbnQ7XG4gIGhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KTtcbiAgbGluZS1oZWlnaHQ6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLWhlaWdodCk7XG4gIGJhY2tncm91bmQ6IHZhcigtLWpwLXdpZGdldHMtaW5wdXQtYmFja2dyb3VuZC1jb2xvcik7XG4gIGNvbG9yOiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LWNvbG9yKTtcbiAgYm9yZGVyOiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LWJvcmRlci13aWR0aCkgc29saWRcbiAgICB2YXIoLS1qcC13aWRnZXRzLWlucHV0LWJvcmRlci1jb2xvcik7XG4gIGZvbnQtc2l6ZTogdmFyKC0tanAtd2lkZ2V0cy1mb250LXNpemUpO1xuICBwYWRkaW5nOiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LXBhZGRpbmcpXG4gICAgY2FsYyh2YXIoLS1qcC13aWRnZXRzLWlucHV0LXBhZGRpbmcpICogMik7XG4gIG1pbi13aWR0aDogMDsgLyogVGhpcyBtYWtlcyBpdCBwb3NzaWJsZSBmb3IgdGhlIGZsZXhib3ggdG8gc2hyaW5rIHRoaXMgaW5wdXQgKi9cbiAgZmxleC1zaHJpbms6IDE7XG4gIGJveC1zaXppbmc6IGJvcmRlci1ib3g7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1jb2xvcnBpY2tlciBpbnB1dFt0eXBlPSd0ZXh0J106ZGlzYWJsZWQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1jb2xvcnBpY2tlciBpbnB1dFt0eXBlPSd0ZXh0J106ZGlzYWJsZWQge1xuICBvcGFjaXR5OiB2YXIoLS1qcC13aWRnZXRzLWRpc2FibGVkLW9wYWNpdHkpO1xufVxuXG4vKiBEYXRlIFBpY2tlciBTdHlsaW5nICovXG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1kYXRlcGlja2VyLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtZGF0ZXBpY2tlciB7XG4gIHdpZHRoOiB2YXIoLS1qcC13aWRnZXRzLWlubGluZS13aWR0aCk7XG4gIGhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KTtcbiAgbGluZS1oZWlnaHQ6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLWhlaWdodCk7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1kYXRlcGlja2VyIGlucHV0W3R5cGU9J2RhdGUnXSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWRhdGVwaWNrZXIgaW5wdXRbdHlwZT0nZGF0ZSddIHtcbiAgZmxleC1ncm93OiAxO1xuICBmbGV4LXNocmluazogMTtcbiAgbWluLXdpZHRoOiAwOyAvKiBUaGlzIG1ha2VzIGl0IHBvc3NpYmxlIGZvciB0aGUgZmxleGJveCB0byBzaHJpbmsgdGhpcyBpbnB1dCAqL1xuICBvdXRsaW5lOiBub25lICFpbXBvcnRhbnQ7XG4gIGhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KTtcbiAgYm9yZGVyOiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LWJvcmRlci13aWR0aCkgc29saWRcbiAgICB2YXIoLS1qcC13aWRnZXRzLWlucHV0LWJvcmRlci1jb2xvcik7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLXdpZGdldHMtaW5wdXQtYmFja2dyb3VuZC1jb2xvcik7XG4gIGNvbG9yOiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LWNvbG9yKTtcbiAgZm9udC1zaXplOiB2YXIoLS1qcC13aWRnZXRzLWZvbnQtc2l6ZSk7XG4gIHBhZGRpbmc6IHZhcigtLWpwLXdpZGdldHMtaW5wdXQtcGFkZGluZylcbiAgICBjYWxjKHZhcigtLWpwLXdpZGdldHMtaW5wdXQtcGFkZGluZykgKiAyKTtcbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWRhdGVwaWNrZXIgaW5wdXRbdHlwZT0nZGF0ZSddOmZvY3VzLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtZGF0ZXBpY2tlciBpbnB1dFt0eXBlPSdkYXRlJ106Zm9jdXMge1xuICBib3JkZXItY29sb3I6IHZhcigtLWpwLXdpZGdldHMtaW5wdXQtZm9jdXMtYm9yZGVyLWNvbG9yKTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWRhdGVwaWNrZXIgaW5wdXRbdHlwZT0nZGF0ZSddOmludmFsaWQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1kYXRlcGlja2VyIGlucHV0W3R5cGU9J2RhdGUnXTppbnZhbGlkIHtcbiAgYm9yZGVyLWNvbG9yOiB2YXIoLS1qcC13YXJuLWNvbG9yMSk7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1kYXRlcGlja2VyIGlucHV0W3R5cGU9J2RhdGUnXTpkaXNhYmxlZCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWRhdGVwaWNrZXIgaW5wdXRbdHlwZT0nZGF0ZSddOmRpc2FibGVkIHtcbiAgb3BhY2l0eTogdmFyKC0tanAtd2lkZ2V0cy1kaXNhYmxlZC1vcGFjaXR5KTtcbn1cblxuLyogUGxheSBXaWRnZXQgKi9cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LXBsYXksIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldC1wbGF5IHtcbiAgd2lkdGg6IHZhcigtLWpwLXdpZGdldHMtaW5saW5lLXdpZHRoLXNob3J0KTtcbiAgZGlzcGxheTogZmxleDtcbiAgYWxpZ24taXRlbXM6IHN0cmV0Y2g7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1wbGF5IC5qdXB5dGVyLWJ1dHRvbiwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LXBsYXkgLmp1cHl0ZXItYnV0dG9uIHtcbiAgZmxleC1ncm93OiAxO1xuICBoZWlnaHQ6IGF1dG87XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1wbGF5IC5qdXB5dGVyLWJ1dHRvbjpkaXNhYmxlZCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LXBsYXkgLmp1cHl0ZXItYnV0dG9uOmRpc2FibGVkIHtcbiAgb3BhY2l0eTogdmFyKC0tanAtd2lkZ2V0cy1kaXNhYmxlZC1vcGFjaXR5KTtcbn1cblxuLyogVGFiIFdpZGdldCAqL1xuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMud2lkZ2V0LXRhYiwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWIge1xuICBkaXNwbGF5OiBmbGV4O1xuICBmbGV4LWRpcmVjdGlvbjogY29sdW1uO1xufVxuXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMud2lkZ2V0LXRhYiA+IC5wLVRhYkJhciwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWIgPiAucC1UYWJCYXIsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLmxtLVRhYkJhciB7XG4gIC8qIE5lY2Vzc2FyeSBzbyB0aGF0IGEgdGFiIGNhbiBiZSBzaGlmdGVkIGRvd24gdG8gb3ZlcmxheSB0aGUgYm9yZGVyIG9mIHRoZSBib3ggYmVsb3cuICovXG4gIG92ZXJmbG93LXg6IHZpc2libGU7XG4gIG92ZXJmbG93LXk6IHZpc2libGU7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy53aWRnZXQtdGFiID4gLnAtVGFiQmFyID4gLnAtVGFiQmFyLWNvbnRlbnQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLy5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLnAtVGFiQmFyID4gLnAtVGFiQmFyLWNvbnRlbnQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLmxtLVRhYkJhciA+IC5sbS1UYWJCYXItY29udGVudCB7XG4gIC8qIE1ha2Ugc3VyZSB0aGF0IHRoZSB0YWIgZ3Jvd3MgZnJvbSBib3R0b20gdXAgKi9cbiAgYWxpZ24taXRlbXM6IGZsZXgtZW5kO1xuICBtaW4td2lkdGg6IDA7XG4gIG1pbi1oZWlnaHQ6IDA7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy53aWRnZXQtdGFiID4gLndpZGdldC10YWItY29udGVudHMsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLndpZGdldC10YWItY29udGVudHMge1xuICB3aWR0aDogMTAwJTtcbiAgYm94LXNpemluZzogYm9yZGVyLWJveDtcbiAgbWFyZ2luOiAwO1xuICBiYWNrZ3JvdW5kOiB2YXIoLS1qcC1sYXlvdXQtY29sb3IxKTtcbiAgY29sb3I6IHZhcigtLWpwLXVpLWZvbnQtY29sb3IxKTtcbiAgYm9yZGVyOiB2YXIoLS1qcC1ib3JkZXItd2lkdGgpIHNvbGlkIHZhcigtLWpwLWJvcmRlci1jb2xvcjEpO1xuICBwYWRkaW5nOiB2YXIoLS1qcC13aWRnZXRzLWNvbnRhaW5lci1wYWRkaW5nKTtcbiAgZmxleC1ncm93OiAxO1xuICBvdmVyZmxvdzogYXV0bztcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLndpZGdldC10YWIgPiAucC1UYWJCYXIsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLy5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLnAtVGFiQmFyLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYiA+IC5sbS1UYWJCYXIge1xuICBmb250OiB2YXIoLS1qcC13aWRnZXRzLWZvbnQtc2l6ZSkgSGVsdmV0aWNhLCBBcmlhbCwgc2Fucy1zZXJpZjtcbiAgbWluLWhlaWdodDogY2FsYyhcbiAgICB2YXIoLS1qcC13aWRnZXRzLWhvcml6b250YWwtdGFiLWhlaWdodCkgKyB2YXIoLS1qcC1ib3JkZXItd2lkdGgpXG4gICk7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy53aWRnZXQtdGFiID4gLnAtVGFiQmFyIC5wLVRhYkJhci10YWIsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLy5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLnAtVGFiQmFyIC5wLVRhYkJhci10YWIsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLmxtLVRhYkJhciAubG0tVGFiQmFyLXRhYiB7XG4gIGZsZXg6IDAgMSB2YXIoLS1qcC13aWRnZXRzLWhvcml6b250YWwtdGFiLXdpZHRoKTtcbiAgbWluLXdpZHRoOiAzNXB4O1xuICBtaW4taGVpZ2h0OiBjYWxjKFxuICAgIHZhcigtLWpwLXdpZGdldHMtaG9yaXpvbnRhbC10YWItaGVpZ2h0KSArIHZhcigtLWpwLWJvcmRlci13aWR0aClcbiAgKTtcbiAgbGluZS1oZWlnaHQ6IHZhcigtLWpwLXdpZGdldHMtaG9yaXpvbnRhbC10YWItaGVpZ2h0KTtcbiAgbWFyZ2luLWxlZnQ6IGNhbGMoLTEgKiB2YXIoLS1qcC1ib3JkZXItd2lkdGgpKTtcbiAgcGFkZGluZzogMHB4IDEwcHg7XG4gIGJhY2tncm91bmQ6IHZhcigtLWpwLWxheW91dC1jb2xvcjIpO1xuICBjb2xvcjogdmFyKC0tanAtdWktZm9udC1jb2xvcjIpO1xuICBib3JkZXI6IHZhcigtLWpwLWJvcmRlci13aWR0aCkgc29saWQgdmFyKC0tanAtYm9yZGVyLWNvbG9yMSk7XG4gIGJvcmRlci1ib3R0b206IG5vbmU7XG4gIHBvc2l0aW9uOiByZWxhdGl2ZTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLndpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYi5wLW1vZC1jdXJyZW50LCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi8uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYiA+IC5wLVRhYkJhciAucC1UYWJCYXItdGFiLnAtbW9kLWN1cnJlbnQsIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLmxtLVRhYkJhciAubG0tVGFiQmFyLXRhYi5sbS1tb2QtY3VycmVudCB7XG4gIGNvbG9yOiB2YXIoLS1qcC11aS1mb250LWNvbG9yMCk7XG4gIC8qIFdlIHdhbnQgdGhlIGJhY2tncm91bmQgdG8gbWF0Y2ggdGhlIHRhYiBjb250ZW50IGJhY2tncm91bmQgKi9cbiAgYmFja2dyb3VuZDogdmFyKC0tanAtbGF5b3V0LWNvbG9yMSk7XG4gIG1pbi1oZWlnaHQ6IGNhbGMoXG4gICAgdmFyKC0tanAtd2lkZ2V0cy1ob3Jpem9udGFsLXRhYi1oZWlnaHQpICsgMiAqIHZhcigtLWpwLWJvcmRlci13aWR0aClcbiAgKTtcbiAgdHJhbnNmb3JtOiB0cmFuc2xhdGVZKHZhcigtLWpwLWJvcmRlci13aWR0aCkpO1xuICBvdmVyZmxvdzogdmlzaWJsZTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLndpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYi5wLW1vZC1jdXJyZW50OmJlZm9yZSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYi5wLW1vZC1jdXJyZW50OmJlZm9yZSwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWIgPiAubG0tVGFiQmFyIC5sbS1UYWJCYXItdGFiLmxtLW1vZC1jdXJyZW50OmJlZm9yZSB7XG4gIHBvc2l0aW9uOiBhYnNvbHV0ZTtcbiAgdG9wOiBjYWxjKC0xICogdmFyKC0tanAtYm9yZGVyLXdpZHRoKSk7XG4gIGxlZnQ6IGNhbGMoLTEgKiB2YXIoLS1qcC1ib3JkZXItd2lkdGgpKTtcbiAgY29udGVudDogJyc7XG4gIGhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1ob3Jpem9udGFsLXRhYi10b3AtYm9yZGVyKTtcbiAgd2lkdGg6IGNhbGMoMTAwJSArIDIgKiB2YXIoLS1qcC1ib3JkZXItd2lkdGgpKTtcbiAgYmFja2dyb3VuZDogdmFyKC0tanAtYnJhbmQtY29sb3IxKTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLndpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYjpmaXJzdC1jaGlsZCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYjpmaXJzdC1jaGlsZCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWIgPiAubG0tVGFiQmFyIC5sbS1UYWJCYXItdGFiOmZpcnN0LWNoaWxkIHtcbiAgbWFyZ2luLWxlZnQ6IDA7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy53aWRnZXQtdGFiXG4gID4gLnAtVGFiQmFyXG4gIC5wLVRhYkJhci10YWI6aG92ZXI6bm90KC5wLW1vZC1jdXJyZW50KSxcbi8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWJcbiAgPiAucC1UYWJCYXJcbiAgLnAtVGFiQmFyLXRhYjpob3Zlcjpub3QoLnAtbW9kLWN1cnJlbnQpLFxuLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy5qdXB5dGVyLXdpZGdldC10YWJcbiAgPiAubG0tVGFiQmFyXG4gIC5sbS1UYWJCYXItdGFiOmhvdmVyOm5vdCgubG0tbW9kLWN1cnJlbnQpIHtcbiAgYmFja2dyb3VuZDogdmFyKC0tanAtbGF5b3V0LWNvbG9yMSk7XG4gIGNvbG9yOiB2YXIoLS1qcC11aS1mb250LWNvbG9yMSk7XG59XG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0cy53aWRnZXQtdGFiXG4gID4gLnAtVGFiQmFyXG4gIC5wLW1vZC1jbG9zYWJsZVxuICA+IC5wLVRhYkJhci10YWJDbG9zZUljb24sXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiXG4+IC5wLVRhYkJhclxuLnAtbW9kLWNsb3NhYmxlXG4+IC5wLVRhYkJhci10YWJDbG9zZUljb24sXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYlxuICA+IC5sbS1UYWJCYXJcbiAgLmxtLW1vZC1jbG9zYWJsZVxuICA+IC5sbS1UYWJCYXItdGFiQ2xvc2VJY29uIHtcbiAgbWFyZ2luLWxlZnQ6IDRweDtcbn1cblxuLyogVGhpcyBmb250LWF3ZXNvbWUgc3RyYXRlZ3kgbWF5IG5vdCB3b3JrIGFjcm9zcyBGQTQgYW5kIEZBNSwgYnV0IHdlIGRvbid0XG5hY3R1YWxseSBzdXBwb3J0IGNsb3NhYmxlIHRhYnMsIHNvIGl0IHJlYWxseSBkb2Vzbid0IG1hdHRlciAqL1xuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLndpZGdldC10YWJcbiAgPiAucC1UYWJCYXJcbiAgLnAtbW9kLWNsb3NhYmxlXG4gID4gLnAtVGFiQmFyLXRhYkNsb3NlSWNvbjpiZWZvcmUsXG4vKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtd2lkZ2V0LXRhYlxuPiAucC1UYWJCYXJcbi5wLW1vZC1jbG9zYWJsZVxuPiAucC1UYWJCYXItdGFiQ2xvc2VJY29uOmJlZm9yZSxcbi8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiXG4gID4gLmxtLVRhYkJhclxuICAubG0tbW9kLWNsb3NhYmxlXG4gID4gLmxtLVRhYkJhci10YWJDbG9zZUljb246YmVmb3JlIHtcbiAgZm9udC1mYW1pbHk6IEZvbnRBd2Vzb21lO1xuICBjb250ZW50OiAnXFxmMDBkJzsgLyogY2xvc2UgKi9cbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXRzLndpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYkljb24sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLyAuanVweXRlci13aWRnZXRzLndpZGdldC10YWIgPiAucC1UYWJCYXIgLnAtVGFiQmFyLXRhYkxhYmVsLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi8gLmp1cHl0ZXItd2lkZ2V0cy53aWRnZXQtdGFiID4gLnAtVGFiQmFyIC5wLVRhYkJhci10YWJDbG9zZUljb24sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi8qIDxERVBSRUNBVEVEPiAqLyAuanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYiA+IC5wLVRhYkJhciAucC1UYWJCYXItdGFiSWNvbiwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovIC5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLnAtVGFiQmFyIC5wLVRhYkJhci10YWJMYWJlbCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovIC5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLnAtVGFiQmFyIC5wLVRhYkJhci10YWJDbG9zZUljb24sIC8qIDwvREVQUkVDQVRFRD4gKi9cbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLmxtLVRhYkJhciAubG0tVGFiQmFyLXRhYkljb24sXG4uanVweXRlci13aWRnZXRzLmp1cHl0ZXItd2lkZ2V0LXRhYiA+IC5sbS1UYWJCYXIgLmxtLVRhYkJhci10YWJMYWJlbCxcbi5qdXB5dGVyLXdpZGdldHMuanVweXRlci13aWRnZXQtdGFiID4gLmxtLVRhYkJhciAubG0tVGFiQmFyLXRhYkNsb3NlSWNvbiB7XG4gIGxpbmUtaGVpZ2h0OiB2YXIoLS1qcC13aWRnZXRzLWhvcml6b250YWwtdGFiLWhlaWdodCk7XG59XG5cbi8qIEFjY29yZGlvbiBXaWRnZXQgKi9cblxuLmp1cHl0ZXItd2lkZ2V0LUNvbGxhcHNlIHtcbiAgZGlzcGxheTogZmxleDtcbiAgZmxleC1kaXJlY3Rpb246IGNvbHVtbjtcbiAgYWxpZ24taXRlbXM6IHN0cmV0Y2g7XG59XG5cbi5qdXB5dGVyLXdpZGdldC1Db2xsYXBzZS1oZWFkZXIge1xuICBwYWRkaW5nOiB2YXIoLS1qcC13aWRnZXRzLWlucHV0LXBhZGRpbmcpO1xuICBjdXJzb3I6IHBvaW50ZXI7XG4gIGNvbG9yOiB2YXIoLS1qcC11aS1mb250LWNvbG9yMik7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLWxheW91dC1jb2xvcjIpO1xuICBib3JkZXI6IHZhcigtLWpwLXdpZGdldHMtYm9yZGVyLXdpZHRoKSBzb2xpZCB2YXIoLS1qcC1ib3JkZXItY29sb3IxKTtcbiAgcGFkZGluZzogY2FsYyh2YXIoLS1qcC13aWRnZXRzLWNvbnRhaW5lci1wYWRkaW5nKSAqIDIgLyAzKVxuICAgIHZhcigtLWpwLXdpZGdldHMtY29udGFpbmVyLXBhZGRpbmcpO1xuICBmb250LXdlaWdodDogYm9sZDtcbn1cblxuLmp1cHl0ZXItd2lkZ2V0LUNvbGxhcHNlLWhlYWRlcjpob3ZlciB7XG4gIGJhY2tncm91bmQtY29sb3I6IHZhcigtLWpwLWxheW91dC1jb2xvcjEpO1xuICBjb2xvcjogdmFyKC0tanAtdWktZm9udC1jb2xvcjEpO1xufVxuXG4uanVweXRlci13aWRnZXQtQ29sbGFwc2Utb3BlbiA+IC5qdXB5dGVyLXdpZGdldC1Db2xsYXBzZS1oZWFkZXIge1xuICBiYWNrZ3JvdW5kLWNvbG9yOiB2YXIoLS1qcC1sYXlvdXQtY29sb3IxKTtcbiAgY29sb3I6IHZhcigtLWpwLXVpLWZvbnQtY29sb3IwKTtcbiAgY3Vyc29yOiBkZWZhdWx0O1xuICBib3JkZXItYm90dG9tOiBub25lO1xufVxuXG4uanVweXRlci13aWRnZXQtQ29sbGFwc2UtY29udGVudHMge1xuICBwYWRkaW5nOiB2YXIoLS1qcC13aWRnZXRzLWNvbnRhaW5lci1wYWRkaW5nKTtcbiAgYmFja2dyb3VuZC1jb2xvcjogdmFyKC0tanAtbGF5b3V0LWNvbG9yMSk7XG4gIGNvbG9yOiB2YXIoLS1qcC11aS1mb250LWNvbG9yMSk7XG4gIGJvcmRlci1sZWZ0OiB2YXIoLS1qcC13aWRnZXRzLWJvcmRlci13aWR0aCkgc29saWQgdmFyKC0tanAtYm9yZGVyLWNvbG9yMSk7XG4gIGJvcmRlci1yaWdodDogdmFyKC0tanAtd2lkZ2V0cy1ib3JkZXItd2lkdGgpIHNvbGlkIHZhcigtLWpwLWJvcmRlci1jb2xvcjEpO1xuICBib3JkZXItYm90dG9tOiB2YXIoLS1qcC13aWRnZXRzLWJvcmRlci13aWR0aCkgc29saWQgdmFyKC0tanAtYm9yZGVyLWNvbG9yMSk7XG4gIG92ZXJmbG93OiBhdXRvO1xufVxuXG4uanVweXRlci13aWRnZXQtQWNjb3JkaW9uIHtcbiAgZGlzcGxheTogZmxleDtcbiAgZmxleC1kaXJlY3Rpb246IGNvbHVtbjtcbiAgYWxpZ24taXRlbXM6IHN0cmV0Y2g7XG59XG5cbi5qdXB5dGVyLXdpZGdldC1BY2NvcmRpb24gLmp1cHl0ZXItd2lkZ2V0LUNvbGxhcHNlIHtcbiAgbWFyZ2luLWJvdHRvbTogMDtcbn1cblxuLmp1cHl0ZXItd2lkZ2V0LUFjY29yZGlvbiAuanVweXRlci13aWRnZXQtQ29sbGFwc2UgKyAuanVweXRlci13aWRnZXQtQ29sbGFwc2Uge1xuICBtYXJnaW4tdG9wOiA0cHg7XG59XG5cbi8qIEhUTUwgd2lkZ2V0ICovXG5cbi8qIDxERVBSRUNBVEVEPiAqL1xuLndpZGdldC1odG1sLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4vKiA8REVQUkVDQVRFRD4gKi8gLndpZGdldC1odG1sbWF0aCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLmp1cHl0ZXItd2lkZ2V0LWh0bWwsXG4uanVweXRlci13aWRnZXQtaHRtbG1hdGgge1xuICBmb250LXNpemU6IHZhcigtLWpwLXdpZGdldHMtZm9udC1zaXplKTtcbn1cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWh0bWwgPiAud2lkZ2V0LWh0bWwtY29udGVudCwgLyogPC9ERVBSRUNBVEVEPiAqL1xuLyogPERFUFJFQ0FURUQ+ICovLndpZGdldC1odG1sbWF0aCA+IC53aWRnZXQtaHRtbC1jb250ZW50LCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtaHRtbCA+IC5qdXB5dGVyLXdpZGdldC1odG1sLWNvbnRlbnQsXG4uanVweXRlci13aWRnZXQtaHRtbG1hdGggPiAuanVweXRlci13aWRnZXQtaHRtbC1jb250ZW50IHtcbiAgLyogRmlsbCBvdXQgdGhlIGFyZWEgaW4gdGhlIEhUTUwgd2lkZ2V0ICovXG4gIGFsaWduLXNlbGY6IHN0cmV0Y2g7XG4gIGZsZXgtZ3JvdzogMTtcbiAgZmxleC1zaHJpbms6IDE7XG4gIC8qIE1ha2VzIHN1cmUgdGhlIGJhc2VsaW5lIGlzIHN0aWxsIGFsaWduZWQgd2l0aCBvdGhlciBlbGVtZW50cyAqL1xuICBsaW5lLWhlaWdodDogdmFyKC0tanAtd2lkZ2V0cy1pbmxpbmUtaGVpZ2h0KTtcbiAgLyogTWFrZSBpdCBwb3NzaWJsZSB0byBoYXZlIGFic29sdXRlbHktcG9zaXRpb25lZCBlbGVtZW50cyBpbiB0aGUgaHRtbCAqL1xuICBwb3NpdGlvbjogcmVsYXRpdmU7XG59XG5cbi8qIEltYWdlIHdpZGdldCAgKi9cblxuLyogPERFUFJFQ0FURUQ+ICovXG4ud2lkZ2V0LWltYWdlLCAvKiA8L0RFUFJFQ0FURUQ+ICovXG4uanVweXRlci13aWRnZXQtaW1hZ2Uge1xuICBtYXgtd2lkdGg6IDEwMCU7XG4gIGhlaWdodDogYXV0bztcbn1cbiJdLCJzb3VyY2VSb290IjoiIn0= */</style><style>/* Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8uLy4uLy4uL3BhY2thZ2VzL2NvbnRyb2xzL2Nzcy93aWRnZXRzLmNzcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7RUFFRSIsInNvdXJjZXNDb250ZW50IjpbIi8qIENvcHlyaWdodCAoYykgSnVweXRlciBEZXZlbG9wbWVudCBUZWFtLlxuICogRGlzdHJpYnV0ZWQgdW5kZXIgdGhlIHRlcm1zIG9mIHRoZSBNb2RpZmllZCBCU0QgTGljZW5zZS5cbiAqL1xuXG5AaW1wb3J0ICcuL2xhYnZhcmlhYmxlcy5jc3MnO1xuQGltcG9ydCAnLi93aWRnZXRzLWJhc2UuY3NzJztcbiJdLCJzb3VyY2VSb290IjoiIn0= */</style><link id="favicon" type="image/x-icon" rel="shortcut icon" href="http://localhost:8888/static/base/images/favicon-notebook.ico"></head>

<body class="notebook_app command_mode" data-jupyter-api-token="4e96327039bdc4c221e33b5c4b840bf1ee43fd997061c1e6" data-base-url="/" data-ws-url="" data-notebook-name="Untitled2.ipynb" data-notebook-path="Untitled2.ipynb" dir="ltr"><div style="visibility: hidden; overflow: hidden; position: absolute; top: 0px; height: 1px; width: auto; padding: 0px; border: 0px; margin: 0px; text-align: left; text-indent: 0px; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal;"><div id="MathJax_Hidden"></div></div><div id="MathJax_Message" style="display: none;"></div>

<noscript>
    <div id='noscript'>
      Jupyter Notebook requires JavaScript.<br>
      Please enable it to proceed. 
  </div>
</noscript>

<div id="header" role="navigation" aria-label="Top Menu" style="display: block;">
  <div id="header-container" class="container">
  <div id="ipython_notebook" class="nav navbar-brand"><a href="http://localhost:8888/tree?token=4e96327039bdc4c221e33b5c4b840bf1ee43fd997061c1e6" title="dashboard">
      <img src="./elandhiya_files/logo.png" alt="Jupyter Notebook">
  </a></div>

  


<span id="save_widget" class="save_widget">
    <span id="notebook_name" class="filename">data type</span>
    <span class="checkpoint_status" title="Tue, Sep 26, 2023 3:34 PM">Last Checkpoint: an hour ago</span>
    <span class="autosave_status">(unsaved changes)</span>
</span>


  

<span id="kernel_logo_widget">
  
  <img class="current_kernel_logo" alt="Current Kernel Logo" src="./elandhiya_files/logo-64x64.png" title="Python 3 (ipykernel)" style="display: inline;">
  
</span>


  
  
  
  

    <span id="login_widget">
      
        <button id="logout" class="btn btn-sm navbar-btn">Logout</button>
      
    </span>

  

  
  
  </div>
  <div class="header-bar"></div>

  
<div id="menubar-container" class="container">
<div id="menubar">
    <div id="menus" class="navbar navbar-default" role="navigation">
        <div class="container-fluid">
            <button type="button" class="btn btn-default navbar-btn navbar-toggle" data-toggle="collapse" data-target=".navbar-collapse">
              <i class="fa fa-bars"></i>
              <span class="navbar-text">Menu</span>
            </button>
            <p id="kernel_indicator" class="navbar-text indicator_area">
              <span class="kernel_indicator_name">Python 3 (ipykernel)</span>
              <i id="kernel_indicator_icon" class="kernel_idle_icon" title="Kernel Idle"></i>
            </p>
            <i id="readonly-indicator" class="navbar-text" title="This notebook is read-only" style="display: none;">
                <span class="fa-stack">
                    <i class="fa fa-save fa-stack-1x"></i>
                    <i class="fa fa-ban fa-stack-2x text-danger"></i>
                </span>
            </i>
            <i id="modal_indicator" class="navbar-text modal_indicator" title="Command Mode"></i>
            <span id="notification_area"><div id="notification_kernel" class="notification_widget btn btn-xs navbar-btn undefined info" style="display: none;"><span></span></div><div id="notification_notebook" class="notification_widget btn btn-xs navbar-btn" style="display: none;"><span></span></div><div id="notification_trusted" class="notification_widget btn btn-xs navbar-btn" style="cursor: help;" disabled="disabled"><span title="Javascript enabled for notebook display">Trusted</span></div><div id="notification_widgets" class="notification_widget btn btn-xs navbar-btn" style="display: none;"><span></span></div></span>
            <div class="navbar-collapse collapse">
              <ul class="nav navbar-nav">
                <li class="dropdown"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="dropdown-toggle" id="filelink" data-toggle="dropdown" aria-haspopup="true" aria-controls="file_menu" aria-expanded="false">File</a>
                    <ul id="file_menu" class="dropdown-menu" role="menu" aria-labelledby="filelink">
                        <li id="new_notebook" class="menu_focus_highlight dropdown dropdown-submenu" role="none">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-haspopup="true" aria-expanded="false" class="dropdown-toggle" data-toggle="dropdown">New Notebook<span class="sr-only">Dropdown</span></a>
                            <ul class="dropdown-menu" id="menu-new-notebook-submenu" role="menu">
                            <li id="new-notebook-submenu-python3"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">Python 3 (ipykernel)</a></li></ul>
                        </li>
                        <li id="open_notebook" role="none" title="Opens a new window with the Dashboard view">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Open...</a></li>
                        <!-- <hr/> -->
                        <li class="divider" role="none"></li>
                        <li id="copy_notebook" role="none" title="Open a copy of this notebook&#39;s contents and start a new kernel">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Make a Copy...</a></li>
                        <li id="save_notebook_as" role="none" title="Save a copy of the notebook&#39;s contents and start a new kernel">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Save as...</a></li>
                        <li id="rename_notebook" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Rename...</a></li>
                        <li id="save_checkpoint" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" class="menu-shortcut-container"><span class="action">Save and Checkpoint</span><span class="kb"><kbd>Ctrl-S</kbd></span></a></li>
                        <!-- <hr/> -->
                        <li class="divider" role="none"></li>
                        <li id="restore_checkpoint" class="menu_focus_highlight dropdown-submenu" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-haspopup="true" aria-expanded="false" class="dropdown-toggle" data-toggle="dropdown">Revert to Checkpoint<span class="sr-only">Dropdown</span></a>
                          <ul class="dropdown-menu"><li><a href="http://localhost:8888/notebooks/data%20type.ipynb#">Tuesday, September 26, 2023 3:34 PM</a></li></ul>
                        </li>
                        <li class="divider" role="none"></li>
                        <li id="print_preview" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Print Preview</a></li>
                        <li class="dropdown-submenu menu_focus_highlight" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-haspopup="true" aria-expanded="false" class="dropdown-toggle" data-toggle="dropdown">Download as<span class="sr-only">Dropdown</span></a>
                            <ul id="download_menu" class="dropdown-menu">
                                
                                <li id="download_asciidoc">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">AsciiDoc (.asciidoc)</a>
                                </li>
                                
                                <li id="download_html">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">HTML (.html)</a>
                                </li>
                                
                                <li id="download_latex">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">LaTeX (.tex)</a>
                                </li>
                                
                                <li id="download_markdown">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">Markdown (.md)</a>
                                </li>
                                
                                <li id="download_notebook">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">Notebook (.ipynb)</a>
                                </li>
                                
                                <li id="download_pdf">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">PDF via LaTeX (.pdf)</a>
                                </li>
                                
                                <li id="download_qtpdf">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">PDF via HTML (.html)</a>
                                </li>
                                
                                <li id="download_qtpng">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">PNG via HTML (.html)</a>
                                </li>
                                
                                <li id="download_rst">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">reST (.rst)</a>
                                </li>
                                
                                <li id="download_script">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">Python (.py)</a>
                                </li>
                                
                                <li id="download_slides">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">Reveal.js slides (.slides.html)</a>
                                </li>
                                
                                <li id="download_webpdf">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">PDF via HTML (.html)</a>
                                </li>
                                
                            </ul>
                        </li>
                        <li class="dropdown-submenu hidden" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Deploy as</a>
                            <ul id="deploy_menu" class="dropdown-menu"></ul>
                        </li>
                        <li class="divider" role="none"></li>
                        <li id="trust_notebook" role="none" title="Trust the output of this notebook" class="disabled">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Trusted Notebook</a></li>
                        <li class="divider" role="none"></li>
                        <li id="close_and_halt" role="none" title="Shutdown this notebook&#39;s kernel, and close this window">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Close and Halt</a></li>
                    </ul>
                </li>

                <li class="dropdown"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="dropdown-toggle" id="editlink" data-toggle="dropdown" aria-haspopup="true" aria-controls="edit_menu">Edit</a>
                    <ul id="edit_menu" class="dropdown-menu" role="menu" aria-labelledby="editlink">
                        <li id="cut_cell" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" class="menu-shortcut-container"><span class="action">Cut Cells</span><span class="kb"><kbd>X</kbd></span></a></li>
                        <li id="copy_cell" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" class="menu-shortcut-container"><span class="action">Copy Cells</span><span class="kb"><kbd>C</kbd></span></a></li>
                        <li id="paste_cell_above" class="disabled" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-disabled="true" class="menu-shortcut-container"><span class="action">Paste Cells Above</span><span class="kb"><kbd>Shift-V</kbd></span></a></li>
                        <li id="paste_cell_below" class="disabled" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-disabled="true" class="menu-shortcut-container"><span class="action">Paste Cells Below</span><span class="kb"><kbd>V</kbd></span></a></li>
                        <li id="paste_cell_replace" class="disabled" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-disabled="true">Paste Cells &amp; Replace</a></li>
                        <li id="delete_cell" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" class="menu-shortcut-container"><span class="action">Delete Cells</span><span class="kb"><kbd>D</kbd>,<kbd>D</kbd></span></a></li>
                        <li id="undelete_cell" class="disabled" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-disabled="true" class="menu-shortcut-container"><span class="action">Undo Delete Cells</span><span class="kb"><kbd>Z</kbd></span></a></li>
                        <li class="divider" role="none"></li>
                        <li id="split_cell" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" class="menu-shortcut-container"><span class="action">Split Cell</span><span class="kb"><kbd>Ctrl-Shift-Minus</kbd></span></a></li>
                        <li id="merge_cell_above" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Merge Cell Above</a></li>
                        <li id="merge_cell_below" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Merge Cell Below</a></li>
                        <li class="divider" role="none"></li>
                        <li id="move_cell_up" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Move Cell Up</a></li>
                        <li id="move_cell_down" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Move Cell Down</a></li>
                        <li class="divider" role="none"></li>
                        <li id="edit_nb_metadata" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Edit Notebook Metadata</a></li>
                        <li class="divider" role="none"></li>
                        <li id="find_and_replace" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem"> Find and Replace </a></li>
                        <li class="divider" role="none"></li>
                        <li id="cut_cell_attachments" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Cut Cell Attachments</a></li>
                        <li id="copy_cell_attachments" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Copy Cell Attachments</a></li>
                        <li id="paste_cell_attachments" class="disabled" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-disabled="true">Paste Cell Attachments</a></li>
                        <li class="divider" role="none"></li>
                        <li id="insert_image" class="disabled" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-disabled="true">  Insert Image </a></li>
                    </ul>
                </li>
                <li class="dropdown"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="dropdown-toggle" id="viewlink" data-toggle="dropdown" aria-haspopup="true" aria-controls="view_menu">View</a>
                    <ul id="view_menu" class="dropdown-menu" role="menu" aria-labelledby="viewlink">
                        <li id="toggle_header" role="none" title="Show/Hide the logo and notebook title (above menu bar)">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Toggle Header</a>
                        </li>
                        <li id="toggle_toolbar" role="none" title="Show/Hide the action icons (below menu bar)">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Toggle Toolbar</a>
                        </li>
                        <li id="toggle_line_numbers" role="none" title="Show/Hide line numbers in cells">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" class="menu-shortcut-container"><span class="action">Toggle Line Numbers</span><span class="kb"><kbd>Shift-L</kbd></span></a>
                        </li>
                        <li id="menu-cell-toolbar" class="menu_focus_highlight dropdown-submenu" role="none">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-haspopup="true" aria-expanded="false" class="dropdown-toggle" data-toggle="dropdown">Cell Toolbar</a>
                            <ul class="dropdown-menu" id="menu-cell-toolbar-submenu"><li data-name="None"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">None</a></li><li data-name="Edit%20Metadata"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">Edit Metadata</a></li><li data-name="Raw%20Cell%20Format"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">Raw Cell Format</a></li><li data-name="Slideshow"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">Slideshow</a></li><li data-name="Attachments"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">Attachments</a></li><li data-name="Tags"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">Tags</a></li></ul>
                        </li>
                    </ul>
                </li>
                <li class="dropdown"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="dropdown-toggle" id="insertlink" data-toggle="dropdown" aria-haspopup="true" aria-controls="insert_menu">Insert</a>
                    <ul id="insert_menu" class="dropdown-menu" role="menu" aria-labelledby="insertlink">
                        <li id="insert_cell_above" role="none" title="Insert an empty Code cell above the currently active cell">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" class="menu-shortcut-container"><span class="action">Insert Cell Above</span><span class="kb"><kbd>A</kbd></span></a></li>
                        <li id="insert_cell_below" role="none" title="Insert an empty Code cell below the currently active cell">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" class="menu-shortcut-container"><span class="action">Insert Cell Below</span><span class="kb"><kbd>B</kbd></span></a></li>
                    </ul>
                </li>
                <li class="dropdown"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="dropdown-toggle" id="celllink" data-toggle="dropdown" aria-haspopup="true" aria-controls="cell_menu">Cell</a>
                    <ul id="cell_menu" class="dropdown-menu" role="menu" aria-labelledby="celllink">
                        <li id="run_cell" role="none" title="Run this cell, and move cursor to the next one">
                            <a role="menuitem" href="http://localhost:8888/notebooks/data%20type.ipynb#" class="menu-shortcut-container"><span class="action">Run Cells</span><span class="kb"><kbd>Ctrl-Enter</kbd></span></a></li>
                        <li id="run_cell_select_below" role="none" title="Run this cell, select below">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" class="menu-shortcut-container"><span class="action">Run Cells and Select Below</span><span class="kb"><kbd>Shift-Enter</kbd></span></a></li>
                        <li id="run_cell_insert_below" role="none" title="Run this cell, insert below">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" class="menu-shortcut-container"><span class="action">Run Cells and Insert Below</span><span class="kb"><kbd>Alt-Enter</kbd></span></a></li>
                        <li id="run_all_cells" role="none" title="Run all cells in the notebook">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Run All</a></li>
                        <li id="run_all_cells_above" role="none" title="Run all cells above (but not including) this cell">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Run All Above</a></li>
                        <li id="run_all_cells_below" role="none" title="Run this cell and all cells below it">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem">Run All Below</a></li>
                        <li class="divider" role="none"></li>
                        <li id="change_cell_type" class="menu_focus_highlight dropdown-submenu" role="none" title="All cells in the notebook have a cell type. By default, new cells are created as &#39;Code&#39; cells">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-haspopup="true" aria-expanded="false" class="dropdown-toggle" data-toggle="dropdown">Cell Type</a>
                            <ul class="dropdown-menu" role="menu">
                              <li id="to_code" role="none" title="Contents will be sent to the kernel for execution, and output will display in the footer of cell">
                                  <a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="menu-shortcut-container"><span class="action">Code</span><span class="kb"><kbd>Y</kbd></span></a></li>
                              <li id="to_markdown" title="Contents will be rendered as HTML and serve as explanatory text">
                                  <a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="menu-shortcut-container"><span class="action">Markdown</span><span class="kb"><kbd>M</kbd></span></a></li>
                              <li id="to_raw" title="Contents will pass through nbconvert unmodified">
                                  <a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="menu-shortcut-container"><span class="action">Raw NBConvert</span><span class="kb"><kbd>R</kbd></span></a></li>
                            </ul>
                        </li>
                        <li class="divider" role="none"></li>
                        <li id="current_outputs" class="menu_focus_highlight dropdown-submenu" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-haspopup="true" aria-expanded="false" class="dropdown-toggle" data-toggle="dropdown">Current Outputs</a>
                            <ul class="dropdown-menu" role="menu">
                                <li id="toggle_current_output" role="none" title="Hide/Show the output of the current cell">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="menu-shortcut-container"><span class="action">Toggle</span><span class="kb"><kbd>O</kbd></span></a>
                                </li>
                                <li id="toggle_current_output_scroll" title="Scroll the output of the current cell">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="menu-shortcut-container"><span class="action">Toggle Scrolling</span><span class="kb"><kbd>Shift-O</kbd></span></a>
                                </li>
                                <li id="clear_current_output" title="Clear the output of the current cell">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">Clear</a>
                                </li>
                            </ul>
                        </li>
                        <li id="all_outputs" class="menu_focus_highlight dropdown-submenu" role="none"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-haspopup="true" aria-expanded="false" class="dropdown-toggle" data-toggle="dropdown">All Output</a>
                            <ul class="dropdown-menu" role="menu">
                                <li id="toggle_all_output" role="none" title="Hide/Show the output of all cells">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">Toggle</a>
                                </li>
                                <li id="toggle_all_output_scroll" title="Scroll the output of all cells">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">Toggle Scrolling</a>
                                </li>
                                <li id="clear_all_output" title="Clear the output of all cells">
                                    <a href="http://localhost:8888/notebooks/data%20type.ipynb#">Clear</a>
                                </li>
                            </ul>
                        </li>
                    </ul>
                </li>
                <li class="dropdown"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="dropdown-toggle" data-toggle="dropdown" id="kernellink">Kernel</a>
                    <ul id="kernel_menu" class="dropdown-menu" aria-labelledby="kernellink">
                        <li id="int_kernel" title="Send Keyboard Interrupt (CTRL-C) to the Kernel">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="menu-shortcut-container"><span class="action">Interrupt</span><span class="kb"><kbd>I</kbd>,<kbd>I</kbd></span></a>
                        </li>
                        <li id="restart_kernel" title="Restart the Kernel">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="menu-shortcut-container"><span class="action">Restart</span><span class="kb"><kbd>0</kbd>,<kbd>0</kbd></span></a>
                        </li>
                        <li id="restart_clear_output" title="Restart the Kernel and clear all output">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#">Restart &amp; Clear Output</a>
                        </li>
                        <li id="restart_run_all" title="Restart the Kernel and re-run the notebook">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#">Restart &amp; Run All</a>
                        </li>
                        <li id="reconnect_kernel" title="Reconnect to the Kernel">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#">Reconnect</a>
                        </li>
                        <li id="shutdown_kernel" title="Shutdown the Kernel">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#">Shutdown</a>
                        </li>
                        <li class="divider" role="none"></li>
                        <li id="menu-change-kernel" class="menu_focus_highlight dropdown-submenu" role="menuitem">
                            <a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="menuitem" aria-haspopup="true" aria-expanded="false" class="dropdown-toggle" data-toggle="dropdown">Change kernel</a>
                            <ul class="dropdown-menu" id="menu-change-kernel-submenu"><li id="kernel-submenu-python3"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">Python 3 (ipykernel)</a></li></ul>
                        </li>
                    </ul>
                </li>
                <li class="dropdown"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" data-toggle="dropdown" class="dropdown-toggle">Widgets</a><ul id="widget-submenu" class="dropdown-menu"><li title="Save the notebook with the widget state information for static rendering"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">Save Notebook Widget State</a></li><li title="Clear the widget state information from the notebook"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">Clear Notebook Widget State</a></li><ul class="divider"></ul><li title="Download the widget state as a JSON file"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">Download Widget State</a></li><li title="Embed interactive widgets"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">Embed Widgets</a></li></ul></li><li class="dropdown"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="dropdown-toggle" data-toggle="dropdown">Help</a>
                    <ul id="help_menu" class="dropdown-menu">
                        
                        <li id="notebook_tour" title="A quick tour of the notebook user interface"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">User Interface Tour</a></li>
                        <li id="keyboard_shortcuts" title="Opens a tooltip with all keyboard shortcuts"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" class="menu-shortcut-container"><span class="action">Keyboard Shortcuts</span><span class="kb"><kbd>H</kbd></span></a></li>
                        <li id="edit_keyboard_shortcuts" title="Opens a dialog allowing you to edit Keyboard shortcuts"><a href="http://localhost:8888/notebooks/data%20type.ipynb#">Edit Keyboard Shortcuts</a></li>
                        <li class="divider"></li>
                        

						
                        
                            
                                <li><a rel="noreferrer" href="http://nbviewer.jupyter.org/github/ipython/ipython/blob/3.x/examples/Notebook/Index.ipynb" target="_blank" title="Opens in a new window">
                                
                                    <i class="fa fa-external-link menu-icon pull-right"></i>
                                

                                Notebook Help
                                </a></li>
                            
                                <li><a rel="noreferrer" href="https://help.github.com/articles/markdown-basics/" target="_blank" title="Opens in a new window">
                                
                                    <i class="fa fa-external-link menu-icon pull-right"></i>
                                

                                Markdown
                                </a></li>
                            
                            
                        
                        <li id="kernel-help-links" class="divider"></li><li><a target="_blank" title="Opens in a new window" href="https://docs.python.org/3.11?v=20230926153353"><i class="fa fa-external-link menu-icon pull-right"></i><span>Python Reference</span></a></li><li><a target="_blank" title="Opens in a new window" href="https://ipython.org/documentation.html?v=20230926153353"><i class="fa fa-external-link menu-icon pull-right"></i><span>IPython Reference</span></a></li><li><a target="_blank" title="Opens in a new window" href="https://docs.scipy.org/doc/numpy/reference/?v=20230926153353"><i class="fa fa-external-link menu-icon pull-right"></i><span>NumPy Reference</span></a></li><li><a target="_blank" title="Opens in a new window" href="https://docs.scipy.org/doc/scipy/reference/?v=20230926153353"><i class="fa fa-external-link menu-icon pull-right"></i><span>SciPy Reference</span></a></li><li><a target="_blank" title="Opens in a new window" href="https://matplotlib.org/contents.html?v=20230926153353"><i class="fa fa-external-link menu-icon pull-right"></i><span>Matplotlib Reference</span></a></li><li><a target="_blank" title="Opens in a new window" href="http://docs.sympy.org/latest/index.html?v=20230926153353"><i class="fa fa-external-link menu-icon pull-right"></i><span>SymPy Reference</span></a></li><li><a target="_blank" title="Opens in a new window" href="https://pandas.pydata.org/pandas-docs/stable/?v=20230926153353"><i class="fa fa-external-link menu-icon pull-right"></i><span>pandas Reference</span></a></li><li class="divider"></li>
                        <li title="About Jupyter Notebook"><a id="notebook_about" href="http://localhost:8888/notebooks/data%20type.ipynb#">About</a></li>
                        
                    </ul>
                </li>
              </ul>
            </div>
        </div>
    </div>
</div>

<div id="maintoolbar" class="navbar">
  <div class="toolbar-inner navbar-inner navbar-nobg">
    <div id="maintoolbar-container" class="container toolbar"><div class="btn-group" id="save-notbook"><button class="btn btn-default" title="Save and Checkpoint" data-jupyter-action="jupyter-notebook:save-notebook"><i class="fa-save fa"></i></button></div><div class="btn-group" id="insert_above_below"><button class="btn btn-default" title="insert cell below" data-jupyter-action="jupyter-notebook:insert-cell-below"><i class="fa-plus fa"></i></button></div><div class="btn-group" id="cut_copy_paste"><button class="btn btn-default" title="cut selected cells" data-jupyter-action="jupyter-notebook:cut-cell"><i class="fa-cut fa"></i></button><button class="btn btn-default" title="copy selected cells" data-jupyter-action="jupyter-notebook:copy-cell"><i class="fa-copy fa"></i></button><button class="btn btn-default" title="paste cells below" data-jupyter-action="jupyter-notebook:paste-cell-below"><i class="fa-paste fa"></i></button></div><div class="btn-group" id="move_up_down"><button class="btn btn-default" title="move selected cells up" data-jupyter-action="jupyter-notebook:move-cell-up"><i class="fa-arrow-up fa"></i></button><button class="btn btn-default" title="move selected cells down" data-jupyter-action="jupyter-notebook:move-cell-down"><i class="fa-arrow-down fa"></i></button></div><div class="btn-group" id="run_int"><button class="btn btn-default" aria-label="Run" title="run cell, select below" data-jupyter-action="jupyter-notebook:run-cell-and-select-next"><i class="fa-play fa"></i><span class="toolbar-btn-label">Run</span></button><button class="btn btn-default" title="interrupt the kernel" data-jupyter-action="jupyter-notebook:interrupt-kernel"><i class="fa-stop fa"></i></button><button class="btn btn-default" title="restart the kernel (with dialog)" data-jupyter-action="jupyter-notebook:confirm-restart-kernel"><i class="fa-repeat fa"></i></button><button class="btn btn-default" title="restart the kernel, then re-run the whole notebook (with dialog)" data-jupyter-action="jupyter-notebook:confirm-restart-kernel-and-run-all-cells"><i class="fa-forward fa"></i></button></div><select id="cell_type" aria-label="combobox, select cell type" role="combobox" class="form-control select-xs"><option value="code">Code</option><option value="markdown">Markdown</option><option value="raw">Raw NBConvert</option><option value="heading">Heading</option><option value="multiselect" disabled="disabled" style="display: none;">-</option></select><div class="btn-group" id="cmd_palette"><button class="btn btn-default" title="open the command palette" data-jupyter-action="jupyter-notebook:show-command-palette"><i class="fa-keyboard-o fa"></i></button></div></div>
  </div>
</div>
</div>

<div class="lower-header-bar"></div>

</div>

<div id="site" style="display: block; height: 817.875px;">


<div id="ipython-main-app">
    <div id="notebook_panel">
        <div id="notebook" tabindex="-1"><div class="container" id="notebook-container"><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[4]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 141.594px; left: 96.4062px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 384.359px; margin-bottom: -15px; border-right-width: 35px; min-height: 164px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre class="CodeMirror-line-like"><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style="visibility: hidden;"><div class="CodeMirror-cursor" style="left: 96.4062px; top: 136px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation" style=""><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">a</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>(<span class="cm-number">42</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">b</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>,<span class="cm-number">5</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">c</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([[<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>],[<span class="cm-number">4</span>,<span class="cm-number">5</span>,<span class="cm-number">6</span>]])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">d</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([[[<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>],[<span class="cm-number">4</span>,<span class="cm-number">5</span>,<span class="cm-number">6</span>]],[[<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>],[<span class="cm-number">4</span>,<span class="cm-number">5</span>,<span class="cm-number">6</span>]]])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">a</span>.<span class="cm-property">ndim</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">b</span>.<span class="cm-property">ndim</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">c</span>.<span class="cm-property">ndim</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">d</span>.<span class="cm-property">ndim</span>)</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 164px;"></div><div class="CodeMirror-gutters" style="display: none; height: 199px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>0
1
2
3
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[5]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 39.5938px; left: 96.4062px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 184.141px; margin-bottom: -15px; border-right-width: 35px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre class="CodeMirror-line-like"><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style=""><div class="CodeMirror-cursor" style="left: 96.4062px; top: 34px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">arr</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">arr</span>[<span class="cm-number">0</span>])</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 97px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>1
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[7]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 39.5938px; left: 127.203px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 184.141px; margin-bottom: -15px; border-right-width: 35px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style=""><div class="CodeMirror-cursor" style="left: 127.203px; top: 34px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">arr</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">arr</span>[<span class="cm-number">2</span>]<span class="cm-operator">+</span><span class="cm-variable">arr</span>[<span class="cm-number">3</span>])</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 97px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>7
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[8]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 39.5938px; left: 304.25px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 322.656px; margin-bottom: -15px; border-right-width: 35px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre class="CodeMirror-line-like"><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style=""><div class="CodeMirror-cursor" style="left: 304.25px; top: 34px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">arr</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([[<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>,<span class="cm-number">5</span>],[<span class="cm-number">6</span>,<span class="cm-number">7</span>,<span class="cm-number">8</span>,<span class="cm-number">9</span>,<span class="cm-number">10</span>]])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-string">'2nd element on 1st row:'</span>,<span class="cm-variable">arr</span>[<span class="cm-number">0</span>,<span class="cm-number">1</span>])</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 97px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>2nd element on 1st row: 2
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[9]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 39.5938px; left: 304.25px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 322.656px; margin-bottom: -15px; border-right-width: 35px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre class="CodeMirror-line-like"><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style=""><div class="CodeMirror-cursor" style="left: 304.25px; top: 34px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">arr</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([[<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>,<span class="cm-number">5</span>],[<span class="cm-number">6</span>,<span class="cm-number">7</span>,<span class="cm-number">8</span>,<span class="cm-number">9</span>,<span class="cm-number">10</span>]])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-string">'5th element on 2nd row:'</span>,<span class="cm-variable">arr</span>[<span class="cm-number">1</span>,<span class="cm-number">4</span>])</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 97px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>5th element on 2nd row: 10
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[10]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 39.5938px; left: 335.047px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 353.453px; margin-bottom: -15px; border-right-width: 35px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre class="CodeMirror-line-like"><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style=""><div class="CodeMirror-cursor" style="left: 335.047px; top: 34px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">arr</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span> ([[<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>,<span class="cm-number">5</span>],[<span class="cm-number">6</span>,<span class="cm-number">7</span>,<span class="cm-number">8</span>,<span class="cm-number">9</span>,<span class="cm-number">10</span>]])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-string">'last element from 2nd dim:'</span>,<span class="cm-variable">arr</span>[<span class="cm-number">1</span>,<span class="cm-operator">-</span><span class="cm-number">1</span>])</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 97px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>last element from 2nd dim: 10
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[20]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 39.5938px; left: 119.516px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 230.359px; margin-bottom: -15px; border-right-width: 35px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style=""><div class="CodeMirror-cursor" style="left: 119.516px; top: 34px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">arr</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>,<span class="cm-number">5</span>,<span class="cm-number">6</span>,<span class="cm-number">7</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">arr</span>[<span class="cm-number">1</span>:<span class="cm-number">5</span>])</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 97px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>[2 3 4 5]
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[21]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 39.5938px; left: 96.3906px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 230.359px; margin-bottom: -15px; border-right-width: 35px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style=""><div class="CodeMirror-cursor" style="left: 96.3906px; top: 34px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">arr</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>,<span class="cm-number">5</span>,<span class="cm-number">6</span>,<span class="cm-number">7</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">arr</span>[<span class="cm-number">4</span>:])</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 97px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>[5 6 7]
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[15]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 39.5938px; left: 96.4062px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 230.359px; margin-bottom: -15px; border-right-width: 35px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style="visibility: hidden;"><div class="CodeMirror-cursor" style="left: 96.4062px; top: 34px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">arr</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>,<span class="cm-number">5</span>,<span class="cm-number">6</span>,<span class="cm-number">7</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">arr</span>[:<span class="cm-number">4</span>])</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 97px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>[1 2 3 4]
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[16]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 39.5938px; left: 119.516px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 238.047px; margin-bottom: -15px; border-right-width: 35px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style="visibility: hidden;"><div class="CodeMirror-cursor" style="left: 119.516px; top: 34px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span>  <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">arr</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span> ([<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>,<span class="cm-number">5</span>,<span class="cm-number">6</span>,<span class="cm-number">7</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">arr</span>[<span class="cm-operator">-</span><span class="cm-number">3</span>:<span class="cm-operator">-</span><span class="cm-number">1</span>])</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 97px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>[5 6]
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[17]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 39.5938px; left: 119.516px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 230.359px; margin-bottom: -15px; border-right-width: 35px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre class="CodeMirror-line-like"><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style="visibility: hidden;"><div class="CodeMirror-cursor" style="left: 119.516px; top: 34px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">arr</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>,<span class="cm-number">5</span>,<span class="cm-number">6</span>,<span class="cm-number">7</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">arr</span>[<span class="cm-number">1</span>:<span class="cm-number">5</span>:<span class="cm-number">2</span>])</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 97px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>[2 4]
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[18]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 39.5938px; left: 119.516px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 315.047px; margin-bottom: -15px; border-right-width: 35px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre class="CodeMirror-line-like"><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style=""><div class="CodeMirror-cursor" style="left: 119.516px; top: 34px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">arr</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([[<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>,<span class="cm-number">5</span>],[<span class="cm-number">6</span>,<span class="cm-number">7</span>,<span class="cm-number">8</span>,<span class="cm-number">9</span>,<span class="cm-number">10</span>]])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">arr</span>[<span class="cm-number">1</span>,<span class="cm-number">1</span>:<span class="cm-number">4</span>])</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 97px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>[7 8 9]
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[19]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 39.5938px; left: 119.516px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 315.047px; margin-bottom: -15px; border-right-width: 35px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre class="CodeMirror-line-like"><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style=""><div class="CodeMirror-cursor" style="left: 119.516px; top: 34px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">arr</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([[<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>,<span class="cm-number">5</span>],[<span class="cm-number">6</span>,<span class="cm-number">7</span>,<span class="cm-number">8</span>,<span class="cm-number">9</span>,<span class="cm-number">10</span>]])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">arr</span>[<span class="cm-number">0</span>:<span class="cm-number">2</span>,<span class="cm-number">2</span>])</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 97px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>[3 8]
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[23]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 22.5938px; left: 27.0938px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 315.047px; margin-bottom: -15px; border-right-width: 35px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre class="CodeMirror-line-like"><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style="visibility: hidden;"><div class="CodeMirror-cursor" style="left: 27.0938px; top: 17px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">arr</span><span class="cm-operator">=</span><span class="cm-variable">np</span>.<span class="cm-property">array</span>([[<span class="cm-number">1</span>,<span class="cm-number">2</span>,<span class="cm-number">3</span>,<span class="cm-number">4</span>,<span class="cm-number">5</span>],[<span class="cm-number">6</span>,<span class="cm-number">7</span>,<span class="cm-number">8</span>,<span class="cm-number">9</span>,<span class="cm-number">10</span>]])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-builtin">print</span>(<span class="cm-variable">arr</span>[<span class="cm-number">0</span>:<span class="cm-number">2</span>,<span class="cm-number">1</span>:<span class="cm-number">4</span>])</span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 97px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout" dir="auto"><pre>[[2 3 4]
 [7 8 9]]
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell unrendered selected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[&nbsp;]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.59375px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 7px; margin-bottom: -15px; border-right-width: 35px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre class="CodeMirror-line-like"><span>xxxxxxxxxx</span></pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style="visibility: hidden;"><div class="CodeMirror-cursor" style="left: 4px; top: 0px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="">​</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 28px;"></div><div class="CodeMirror-gutters" style="display: none; height: 63px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div><div class="cell code_cell unrendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[&nbsp;]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.59375px; left: 4px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" tabindex="-1" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" tabindex="-1" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 7px; margin-bottom: -15px; border-right-width: 35px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style="visibility: hidden;"><div class="CodeMirror-cursor" style="left: 4px; top: 0px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="">​</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 35px; width: 1px; border-bottom: 0px solid transparent; top: 28px;"></div><div class="CodeMirror-gutters" style="display: none; height: 63px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to expand output; double click to hide output"></div><div class="output"></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div></div><div class="end_space"></div></div>
        <div id="tooltip" class="ipython_tooltip" style="display:none"><div class="tooltipbuttons"><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="button" class="ui-button"><span class="ui-icon ui-icon-close">Close</span></a><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="button" class="ui-button" id="expanbutton" title="Grow the tooltip vertically (press shift-tab twice)"><span class="ui-icon ui-icon-plus">Expand</span></a><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="button" class="ui-button" title="show the current docstring in pager (press shift-tab 4 times)"><span class="ui-icon ui-icon-arrowstop-l-n">Open in Pager</span></a><a href="http://localhost:8888/notebooks/data%20type.ipynb#" role="button" class="ui-button" title="Tooltip will linger for 10 seconds while you type" style="display: none;"><span class="ui-icon ui-icon-clock">Close</span></a></div><div class="pretooltiparrow"></div><div class="tooltiptext smalltooltip"></div></div>
    </div>
</div>



</div>



<div id="pager" class="ui-resizable">
    <div id="pager-contents">
        <div id="pager-container" class="container"></div>
    </div>
    <div id="pager-button-area"><a role="button" title="Open the pager in an external window" class="ui-button"><span class="ui-icon ui-icon-extlink"></span></a><a role="button" title="Close the pager" class="ui-button"><span class="ui-icon ui-icon-close"></span></a></div>
<div class="ui-resizable-handle ui-resizable-n" style="z-index: 90;"></div></div>






<script type="text/javascript">
    sys_info = {"notebook_version": "6.5.2", "notebook_path": "C:\\Users\\admin\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\notebook", "commit_source": "", "commit_hash": "", "sys_version": "3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)]", "sys_executable": "C:\\Users\\admin\\AppData\\Local\\Programs\\Python\\Python311\\python.exe", "sys_platform": "win32", "platform": "Windows-10-10.0.19044-SP0", "os_name": "nt", "default_encoding": "utf-8"};
</script>

<script src="./elandhiya_files/encoding.js.download" charset="utf-8"></script>

<script src="./elandhiya_files/main.min.js.download" type="text/javascript" charset="utf-8"></script>



<script type="text/javascript">
  function _remove_token_from_url() {
    if (window.location.search.length <= 1) {
      return;
    }
    var search_parameters = window.location.search.slice(1).split('&');
    for (var i = 0; i < search_parameters.length; i++) {
      if (search_parameters[i].split('=')[0] === 'token') {
        // remote token from search parameters
        search_parameters.splice(i, 1);
        var new_search = '';
        if (search_parameters.length) {
          new_search = '?' + search_parameters.join('&');
        }
        var new_url = window.location.origin + 
                      window.location.pathname + 
                      new_search + 
                      window.location.hash;
        window.history.replaceState({}, "", new_url);
        return;
      }
    }
  }
  _remove_token_from_url();
</script>


<div style="position: absolute; width: 0px; height: 0px; overflow: hidden; padding: 0px; border: 0px; margin: 0px;"><div id="MathJax_Font_Test" style="position: absolute; visibility: hidden; top: 0px; left: 0px; width: auto; min-width: 0px; max-width: none; padding: 0px; border: 0px; margin: 0px; white-space: nowrap; text-align: left; text-indent: 0px; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; font-size: 40px; font-weight: normal; font-style: normal;"></div></div><div class="modal cmd-palette" style="display: none;"><div class="modal-dialog"><div class="modal-content"><div class="modal-body"><form><div class="typeahead__container"><div class="typeahead__field"><span class="typeahead__query" style="position: relative;"><span class="typeahead__cancel-button">×</span><input type="search"><input type="search" readonly="readonly" unselectable="on" aria-hidden="true" tabindex="-1" class="typeahead__hint" style="border-color: transparent; position: absolute; top: 0px; display: inline; z-index: -1; float: none; color: rgb(192, 192, 192); box-shadow: none; cursor: default; user-select: none;"></span><span class="typeahead__button"><button type="submit"><span class="typeahead__search-icon"></span></button></span></div><div class="typeahead__result"><ul class="typeahead__list"><li class="typeahead__group" data-search-group="jupyter-notebook"><a href="javascript:;" tabindex="-1">jupyter-notebook command group</a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="0"><a href="javascript:;"><i class="fa fa-icon "></i>automatically indent selection  <div title="jupyter-notebook:auto-indent" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="1"><a href="javascript:;"><i class="fa fa-icon "></i>change cell to code  <div title="jupyter-notebook:change-cell-to-code" class="pull-right command-shortcut"><kbd>Y</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="2"><a href="javascript:;"><i class="fa fa-icon "></i>change cell to heading 1  <div title="jupyter-notebook:change-cell-to-heading-1" class="pull-right command-shortcut"><kbd>1</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="3"><a href="javascript:;"><i class="fa fa-icon "></i>change cell to heading 2  <div title="jupyter-notebook:change-cell-to-heading-2" class="pull-right command-shortcut"><kbd>2</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="4"><a href="javascript:;"><i class="fa fa-icon "></i>change cell to heading 3  <div title="jupyter-notebook:change-cell-to-heading-3" class="pull-right command-shortcut"><kbd>3</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="5"><a href="javascript:;"><i class="fa fa-icon "></i>change cell to heading 4  <div title="jupyter-notebook:change-cell-to-heading-4" class="pull-right command-shortcut"><kbd>4</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="6"><a href="javascript:;"><i class="fa fa-icon "></i>change cell to heading 5  <div title="jupyter-notebook:change-cell-to-heading-5" class="pull-right command-shortcut"><kbd>5</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="7"><a href="javascript:;"><i class="fa fa-icon "></i>change cell to heading 6  <div title="jupyter-notebook:change-cell-to-heading-6" class="pull-right command-shortcut"><kbd>6</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="8"><a href="javascript:;"><i class="fa fa-icon "></i>change cell to markdown  <div title="jupyter-notebook:change-cell-to-markdown" class="pull-right command-shortcut"><kbd>M</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="9"><a href="javascript:;"><i class="fa fa-icon "></i>change cell to raw  <div title="jupyter-notebook:change-cell-to-raw" class="pull-right command-shortcut"><kbd>R</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="10"><a href="javascript:;"><i class="fa fa-icon "></i>clear all cells output  <div title="jupyter-notebook:clear-all-cells-output" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="11"><a href="javascript:;"><i class="fa fa-icon "></i>clear cell output  <div title="jupyter-notebook:clear-cell-output" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="12"><a href="javascript:;"><i class="fa fa-icon "></i>close the pager  <div title="jupyter-notebook:close-pager" class="pull-right command-shortcut"><kbd>Esc</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="13"><a href="javascript:;"><i class="fa fa-icon fa-repeat"></i>confirm restart kernel  <div title="jupyter-notebook:confirm-restart-kernel" class="pull-right command-shortcut"><kbd>0</kbd>,<kbd>0</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="14"><a href="javascript:;"><i class="fa fa-icon "></i>confirm restart kernel and clear output  <div title="jupyter-notebook:confirm-restart-kernel-and-clear-output" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="15"><a href="javascript:;"><i class="fa fa-icon fa-forward"></i>confirm restart kernel and run all cells  <div title="jupyter-notebook:confirm-restart-kernel-and-run-all-cells" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="16"><a href="javascript:;"><i class="fa fa-icon fa-repeat"></i>confirm shutdown kernel  <div title="jupyter-notebook:confirm-shutdown-kernel" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="17"><a href="javascript:;"><i class="fa fa-icon "></i>copy cell attachments  <div title="jupyter-notebook:copy-cell-attachments" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="18"><a href="javascript:;"><i class="fa fa-icon fa-copy"></i>copy selected cells  <div title="jupyter-notebook:copy-cell" class="pull-right command-shortcut"><kbd>C</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="19"><a href="javascript:;"><i class="fa fa-icon "></i>cut cell attachments  <div title="jupyter-notebook:cut-cell-attachments" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="20"><a href="javascript:;"><i class="fa fa-icon fa-cut"></i>cut selected cells  <div title="jupyter-notebook:cut-cell" class="pull-right command-shortcut"><kbd>X</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="21"><a href="javascript:;"><i class="fa fa-icon "></i>delete cells  <div title="jupyter-notebook:delete-cell" class="pull-right command-shortcut"><kbd>D</kbd>,<kbd>D</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="22"><a href="javascript:;"><i class="fa fa-icon "></i>duplicate notebook  <div title="jupyter-notebook:duplicate-notebook" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="23"><a href="javascript:;"><i class="fa fa-icon "></i>edit command mode keyboard shortcuts  <div title="jupyter-notebook:edit-command-mode-keyboard-shortcuts" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="24"><a href="javascript:;"><i class="fa fa-icon "></i>enter command mode  <div title="jupyter-notebook:enter-command-mode" class="pull-right edit-shortcut"><kbd>Esc</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="25"><a href="javascript:;"><i class="fa fa-icon "></i>enter edit mode  <div title="jupyter-notebook:enter-edit-mode" class="pull-right command-shortcut"><kbd>Enter</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="26"><a href="javascript:;"><i class="fa fa-icon "></i>extend selection above  <div title="jupyter-notebook:extend-selection-above" class="pull-right command-shortcut"><kbd>Shift-K</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="27"><a href="javascript:;"><i class="fa fa-icon "></i>extend selection below  <div title="jupyter-notebook:extend-selection-below" class="pull-right command-shortcut"><kbd>Shift-J</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="28"><a href="javascript:;"><i class="fa fa-icon "></i>find and replace  <div title="jupyter-notebook:find-and-replace" class="pull-right command-shortcut"><kbd>F</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="29"><a href="javascript:;"><i class="fa fa-icon "></i>hide all line numbers  <div title="jupyter-notebook:hide-all-line-numbers" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="30"><a href="javascript:;"><i class="fa fa-icon "></i>hide menubar  <div title="jupyter-notebook:hide-menubar" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="31"><a href="javascript:;"><i class="fa fa-icon "></i>hide the header  <div title="jupyter-notebook:hide-header" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="32"><a href="javascript:;"><i class="fa fa-icon "></i>hide the toolbar  <div title="jupyter-notebook:hide-toolbar" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="33"><a href="javascript:;"><i class="fa fa-icon "></i>ignore  <div title="jupyter-notebook:ignore" class="pull-right command-shortcut"><kbd>Shift</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="34"><a href="javascript:;"><i class="fa fa-icon "></i>insert cell above  <div title="jupyter-notebook:insert-cell-above" class="pull-right command-shortcut"><kbd>A</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="35"><a href="javascript:;"><i class="fa fa-icon fa-plus"></i>insert cell below  <div title="jupyter-notebook:insert-cell-below" class="pull-right command-shortcut"><kbd>B</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="36"><a href="javascript:;"><i class="fa fa-icon "></i>insert image  <div title="jupyter-notebook:insert-image" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="37"><a href="javascript:;"><i class="fa fa-icon fa-stop"></i>interrupt the kernel  <div title="jupyter-notebook:interrupt-kernel" class="pull-right command-shortcut"><kbd>I</kbd>,<kbd>I</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="38"><a href="javascript:;"><i class="fa fa-icon "></i>merge cell with next cell  <div title="jupyter-notebook:merge-cell-with-next-cell" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="39"><a href="javascript:;"><i class="fa fa-icon "></i>merge cell with previous cell  <div title="jupyter-notebook:merge-cell-with-previous-cell" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="40"><a href="javascript:;"><i class="fa fa-icon "></i>merge cells  <div title="jupyter-notebook:merge-cells" class="pull-right command-shortcut"><kbd>Shift-M</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="41"><a href="javascript:;"><i class="fa fa-icon "></i>merge selected cells  <div title="jupyter-notebook:merge-selected-cells" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="42"><a href="javascript:;"><i class="fa fa-icon fa-arrow-down"></i>move cells down  <div title="jupyter-notebook:move-cell-down" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="43"><a href="javascript:;"><i class="fa fa-icon fa-arrow-up"></i>move cells up  <div title="jupyter-notebook:move-cell-up" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="44"><a href="javascript:;"><i class="fa fa-icon "></i>move cursor down  <div title="jupyter-notebook:move-cursor-down" class="pull-right edit-shortcut"><kbd>Down</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="45"><a href="javascript:;"><i class="fa fa-icon "></i>move cursor up  <div title="jupyter-notebook:move-cursor-up" class="pull-right edit-shortcut"><kbd>Up</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="46"><a href="javascript:;"><i class="fa fa-icon "></i>paste cell attachments  <div title="jupyter-notebook:paste-cell-attachments" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="47"><a href="javascript:;"><i class="fa fa-icon "></i>paste cell replace  <div title="jupyter-notebook:paste-cell-replace" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="48"><a href="javascript:;"><i class="fa fa-icon "></i>paste cells above  <div title="jupyter-notebook:paste-cell-above" class="pull-right command-shortcut"><kbd>Shift-V</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="49"><a href="javascript:;"><i class="fa fa-icon fa-paste"></i>paste cells below  <div title="jupyter-notebook:paste-cell-below" class="pull-right command-shortcut"><kbd>V</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="50"><a href="javascript:;"><i class="fa fa-icon "></i>rename notebook  <div title="jupyter-notebook:rename-notebook" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="51"><a href="javascript:;"><i class="fa fa-icon "></i>restart kernel  <div title="jupyter-notebook:restart-kernel" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="52"><a href="javascript:;"><i class="fa fa-icon "></i>restart kernel and clear output  <div title="jupyter-notebook:restart-kernel-and-clear-output" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="53"><a href="javascript:;"><i class="fa fa-icon "></i>restart kernel and run all cells  <div title="jupyter-notebook:restart-kernel-and-run-all-cells" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="54"><a href="javascript:;"><i class="fa fa-icon "></i>run all cells  <div title="jupyter-notebook:run-all-cells" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="55"><a href="javascript:;"><i class="fa fa-icon "></i>run all cells above  <div title="jupyter-notebook:run-all-cells-above" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="56"><a href="javascript:;"><i class="fa fa-icon "></i>run all cells below  <div title="jupyter-notebook:run-all-cells-below" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="57"><a href="javascript:;"><i class="fa fa-icon "></i>run cell and insert below  <div title="jupyter-notebook:run-cell-and-insert-below" class="pull-right command-shortcut"><kbd>Alt-Enter</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="58"><a href="javascript:;"><i class="fa fa-icon fa-play"></i>run cell and select next  <div title="jupyter-notebook:run-cell-and-select-next" class="pull-right command-shortcut"><kbd>Shift-Enter</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="59"><a href="javascript:;"><i class="fa fa-icon "></i>run selected cells  <div title="jupyter-notebook:run-cell" class="pull-right command-shortcut"><kbd>Ctrl-Enter</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="60"><a href="javascript:;"><i class="fa fa-icon fa-save"></i>save notebook  <div title="jupyter-notebook:save-notebook" class="pull-right command-shortcut"><kbd>Ctrl-S</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="61"><a href="javascript:;"><i class="fa fa-icon "></i>scroll cell center  <div title="jupyter-notebook:scroll-cell-center" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="62"><a href="javascript:;"><i class="fa fa-icon "></i>scroll cell top  <div title="jupyter-notebook:scroll-cell-top" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="63"><a href="javascript:;"><i class="fa fa-icon "></i>scroll notebook down  <div title="jupyter-notebook:scroll-notebook-down" class="pull-right command-shortcut"><kbd>Space</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="64"><a href="javascript:;"><i class="fa fa-icon "></i>scroll notebook up  <div title="jupyter-notebook:scroll-notebook-up" class="pull-right command-shortcut"><kbd>Shift-Space</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="65"><a href="javascript:;"><i class="fa fa-icon "></i>select all  <div title="jupyter-notebook:select-all" class="pull-right command-shortcut"><kbd>Ctrl-A</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="66"><a href="javascript:;"><i class="fa fa-icon "></i>select next cell  <div title="jupyter-notebook:select-next-cell" class="pull-right command-shortcut"><kbd>Down</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="67"><a href="javascript:;"><i class="fa fa-icon "></i>select previous cell  <div title="jupyter-notebook:select-previous-cell" class="pull-right command-shortcut"><kbd>Up</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="68"><a href="javascript:;"><i class="fa fa-icon "></i>show all line numbers  <div title="jupyter-notebook:show-all-line-numbers" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="69"><a href="javascript:;"><i class="fa fa-icon fa-keyboard-o"></i>show command pallette  <div title="jupyter-notebook:show-command-palette" class="pull-right command-shortcut"><kbd>Ctrl-Shift-P</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="70"><a href="javascript:;"><i class="fa fa-icon "></i>show keyboard shortcuts  <div title="jupyter-notebook:show-keyboard-shortcuts" class="pull-right command-shortcut"><kbd>H</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="71"><a href="javascript:;"><i class="fa fa-icon "></i>show menubar  <div title="jupyter-notebook:show-menubar" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="72"><a href="javascript:;"><i class="fa fa-icon "></i>show the header  <div title="jupyter-notebook:show-header" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="73"><a href="javascript:;"><i class="fa fa-icon "></i>show the toolbar  <div title="jupyter-notebook:show-toolbar" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="74"><a href="javascript:;"><i class="fa fa-icon "></i>shutdown kernel  <div title="jupyter-notebook:shutdown-kernel" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="75"><a href="javascript:;"><i class="fa fa-icon "></i>shutdown kernel and close window  <div title="jupyter-notebook:close-and-halt" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="76"><a href="javascript:;"><i class="fa fa-icon "></i>split cell at cursor(s)  <div title="jupyter-notebook:split-cell-at-cursor" class="pull-right edit-shortcut"><kbd>Ctrl-Shift-Minus</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="77"><a href="javascript:;"><i class="fa fa-icon "></i>toggle all cells output collapsed  <div title="jupyter-notebook:toggle-all-cells-output-collapsed" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="78"><a href="javascript:;"><i class="fa fa-icon "></i>toggle all cells output scrolled  <div title="jupyter-notebook:toggle-all-cells-output-scrolled" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="79"><a href="javascript:;"><i class="fa fa-icon fa-list-ol"></i>toggle all line numbers  <div title="jupyter-notebook:toggle-all-line-numbers" class="pull-right command-shortcut"><kbd>Shift-L</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="80"><a href="javascript:;"><i class="fa fa-icon "></i>toggle cell output  <div title="jupyter-notebook:toggle-cell-output-collapsed" class="pull-right command-shortcut"><kbd>O</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="81"><a href="javascript:;"><i class="fa fa-icon "></i>toggle cell scrolling  <div title="jupyter-notebook:toggle-cell-output-scrolled" class="pull-right command-shortcut"><kbd>Shift-O</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="82"><a href="javascript:;"><i class="fa fa-icon "></i>toggle current cell ltr/rtl direction  <div title="jupyter-notebook:toggle-cell-rtl-layout" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="83"><a href="javascript:;"><i class="fa fa-icon "></i>toggle header  <div title="jupyter-notebook:toggle-header" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="84"><a href="javascript:;"><i class="fa fa-icon "></i>toggle line numbers  <div title="jupyter-notebook:toggle-cell-line-numbers" class="pull-right command-shortcut"><kbd>L</kbd></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="85"><a href="javascript:;"><i class="fa fa-icon "></i>toggle menubar  <div title="jupyter-notebook:toggle-menubar" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="86"><a href="javascript:;"><i class="fa fa-icon "></i>toggle notebook ltr/rtl direction  <div title="jupyter-notebook:toggle-rtl-layout" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="87"><a href="javascript:;"><i class="fa fa-icon "></i>toggle toolbar  <div title="jupyter-notebook:toggle-toolbar" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="88"><a href="javascript:;"><i class="fa fa-icon "></i>trust notebook  <div title="jupyter-notebook:trust-notebook" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-jupyter-notebook" data-group="jupyter-notebook" data-index="89"><a href="javascript:;"><i class="fa fa-icon "></i>undo cell deletion  <div title="jupyter-notebook:undo-cell-deletion" class="pull-right command-shortcut"><kbd>Z</kbd></div></a></li><li class="typeahead__group" data-search-group="widgets"><a href="javascript:;" tabindex="-1">widgets command group</a></li><li class="typeahead__item typeahead__group-widgets" data-group="widgets" data-index="90"><a href="javascript:;"><i class="fa fa-icon fa-sliders"></i>embed interactive widgets  <div title="widgets:embed-interactive-widgets" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-widgets" data-group="widgets" data-index="91"><a href="javascript:;"><i class="fa fa-icon fa-truck"></i>save clear widgets  <div title="widgets:save-clear-widgets" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-widgets" data-group="widgets" data-index="92"><a href="javascript:;"><i class="fa fa-icon fa-sliders"></i>save widget state  <div title="widgets:save-widget-state" class="pull-right no-shortcut"></div></a></li><li class="typeahead__item typeahead__group-widgets" data-group="widgets" data-index="93"><a href="javascript:;"><i class="fa fa-icon fa-truck"></i>save with widgets  <div title="widgets:save-with-widgets" class="pull-right no-shortcut"></div></a></li></ul></div></div></form></div></div></div></div></body></html>