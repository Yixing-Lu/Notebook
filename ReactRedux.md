# Basic

## Components

when state change, React create new Virtual Dom, compare to old one

Components looks like templates, actually JSX

Contains state: data and UI state and JS functions



Use className instead of class=""

onClick, onMouseOver, onCopy, onChange

onSubmit: capture enter and click button, `e.preventDefault();`



**this** keyword in function determined by how and where this func is called, not where the func sits

use arrow func to bind **this** to component: `handleClick = (e) => {}`

change state: `this.setState({ name: "Yoshi"})`



**Anonymous Function**

```jsx
<button
  onClick={() => {
    deleteNinja(ninja.id);
  }}
  >
  Delete Ninja
</button>
```



**Add CSS**

`import "./Ninjas.css";`



## props

 pass data from parent component to child component

`<Ninjas name="Ryu" age="25" belt="black"/>`

`this.props`



## Destructuring

`const {name, age, belt} = this.props;`



## Map

```jsx
const ninjaList = ninjas.map((ninja) => {
      return (
        <div className="ninja" key={ninja.id}>
          <div>Name: {ninja.name}</div>
          <div>Age: {ninja.age}</div>
          <div>Belt: {ninja.belt}</div>
        </div>
      );
    });
```



## Container & UI components

Container: contain state, lifecycle hooks, use classes to create, not concerned with UI

UI(stateless & functional): not contain state, receive data from props, only concern UI, use functions

```jsx
const Ninjas = (props) => {
  const { ninjas } = props;
  const ninjaList = ninjas.map((ninja) => {
    return (
      <div className="ninja" key={ninja.id}>
      </div>
    );
  });
  return <div className="ninja-list">{ninjaList}</div>;
};
```

```jsx
const Ninjas = ({ ninjas, xxx, yyy }) => {
  const ninjaList = ninjas.map((ninja) => {
    return (
      <div className="ninja" key={ninja.id}>
      </div>
    );
  });
  return <div className="ninja-list">{ninjaList}</div>;
};
```



## Functions as Props

```jsx
Parent:
<AddNinja addNinja={this.addNinja} />

Child:
handleSubmit = (e) => {
  e.preventDefault();
  this.props.addNinja(this.state);
};
```



## Update state

**Append element into list in state**

```jsx
addNinja = (newNinja) => {
    newNinja.id = Math.random();
    let ninjasList = [...this.state.ninjas, newNinja];
    this.setState({
      ninjas: ninjasList,
    });
  };
```



**filter**

```jsx
  deleteNinja = (id) => {
    let ninjas = this.state.ninjas.filter((ninja) => {
      return ninja.id !== id;
    });
    this.setState({
      ninjas: ninjas,
    });
  };
```



## Lifecycle

Mount -> Update -> Unmount

constructor	render()

componentDidMount()

componentDidUpdate()

componentWillUnmount()





# Router

## Routes

stop request from going to server, react intercept, eject different components

`npm install react-router-dom`

```jsx
 render() {
    return (
      <BrowserRouter>
        <div className="App">
          <Navbar />
          <Route exact path="/" component={Home}></Route>
          <Route path="/about" component={About}></Route>
          <Route path="/contact" component={Contact}></Route>
        </div>
      </BrowserRouter>
    );
  }
```



## Link, NavLink

```jsx
import { Link, NavLink } from "react-router-dom";
<li>
  <Link to="/">Home</Link>
</li>
<li>
  <Link to="/about">About</Link>
</li>
<li>
  <Link to="/contact">Coontact</Link>
</li>
```

NavLink: add class="active"



## Redirects

Route in BrowswerRoute will have props

```jsx
  setTimeout(() => {
    props.history.push("/about");
  }, 2000);
```



## Higher order component

Add func to supercharged

`export default withRouter(Navbar);`

```jsx
const Rainbow = (WrappedComponent) => {
  const colours = ["red", "pink", "green", "orange"];
  const randomColor = colours[Math.floor(Math.random() * 3)];
  const className = randomColor + "-text";
  return (props) => {
    return (
      <div className={className}>
        <WrappedComponent {...props} />
      </div>
    );
  };
};

export default Rainbow;

----------

Rainbow(About);
```



## Axios

`npm install axios`

Lifecycle hooks, use class based component, at componentDidMount()

```jsx
import axios from "axios";

componentDidMount() {
  axios.get("https://jsonplaceholder.typicode.com/posts").then((res) => {
    this.setState({
      posts: res.data.slice(0, 10),
    });
  });
}
```





## Params

```jsx
Home.js
<Link to={"/" + post.id}>
  <span className="card-title">{post.title}</span>
</Link>
  
----------

App.js:
<Route path="/:post_id" component={Post} />

Post.js:
componentDidMount() {
  let id = this.props.match.params.post_id;
  axios
    .get("https://jsonplaceholder.typicode.com/posts/" + id)
    .then((res) => {
    this.setState({
      post: res.data,
    });
  });
}

----------

render() {
  const post = this.state.post ? (
    <div className="post">
      <h4 className="center">{this.state.post.title}</h4>
      <p>{this.state.post.body}</p>
    </div>
  ) : (
    <div className="center">Loading</div>
  );
  return <div className="container">{post}</div>;
}
```



## Switch

only first one route match and stop

```jsx
import { BrowserRouter, Route, Switch } from "react-router-dom";

render() {
    return (
      <BrowserRouter>
        <div className="App">
          <Navbar />
          <Switch>
            <Route exact path="/" component={Home} />
            <Route path="/about" component={About} />
            <Route path="/contact" component={Contact} />
            <Route path="/:post_id" component={Post} />
          </Switch>
        </div>
      </BrowserRouter>
    );
  }
```





# Redux

central data store for all app data

any component can access data

**Redux(central store)** -> component subscribes to change via props ->

**Component** -> component dispatched an action with optional payload -> 

**Dispatch Action** -> action passed to Reducer -> 

**Reducer** -> reducer updates the central state



**Store**

Create store

```jsx
const {createStore} = Redux;
const initState = {
  todos: [],
  post: []
}
function myreducer(state = initState, action) {
}
const store = createStore(myreducer);
store.subscribe(() => {
  console.log(store.getState())
})
```



**Action**

```jsx
const todoAction = {type:"ADD_TODO", todo: "buy milk"}
store.dispatch(todoAction) // dispatch
```



**Reducer**

```jsx
function myreducer(state = initState, action) {
  if (action.type == "ADD_TODO"){
    return {
      ...state,
      todos: [...state.todos, action.todo]
    }
  }
}
```



## Setup

`npm install redux react-redux`

```jsx
index.js
------------------
import { createStore } from "redux";
import { Provider } from "react-redux";
import rootReducer from "./reducers/rootReducer";

const store = createStore(rootReducer);

ReactDOM.render(
  <React.StrictMode>
    <Provider store={store}>
      <App />
    </Provider>
  </React.StrictMode>,
  document.getElementById("root")
);
```



Create reducers in /src/reducers

```jsx
rootReducer.js
------------------
const initState = {
  posts: [],
};
const rootReducer = (state = initState, action) => {
  return state;
};

export default rootReducer;
```

```jsx
Home.js
------------------
import { connect } from "react-redux";

const mapStateToProps = (state) => {
  return {
    posts: state.posts,
  };
};

export default connect(mapStateToProps)(Home);
```

```jsx
Post.js
------------------
handleClick = () => {
  this.props.deletePost(this.props.post.id);
  this.props.history.push("/");
};

<button className="btn grey" onClick={this.props.handleClick}>
  Delete Post
</button>

const mapStateToProps = (state, ownProps) => {
  let id = ownProps.match.params.post_id;
  return {
    post: state.posts.find((post) => post.id === id),
  };
};

const mapDispatchToProps = (dispatch) => {
  return {
    deletePost: (id) => {
      dispatch(deletePost(id));
    },
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(Post);
```



**Action creators**

Create actions in /src/actions/postActions.js

```jsx
export const deletePost = (id) => {
  return {
    type: "DELETE_POST",
    id,
  };
};
```





